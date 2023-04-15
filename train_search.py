import os
import sys
import time
import glob
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from datasets import load_data, cal_diameter
from model_search import Network
import numpy as np
from ogb.graphproppred import Evaluator
from logging_util import init_logger
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK, rand
from genotypes import SC_PRIMITIVES, NA_PRIMITIVES, FF_PRIMITIVES
from cal_range import cal_range

parser = argparse.ArgumentParser("")
parser.add_argument('--data', type=str, default='DD', help='dataset name')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0005, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay')
parser.add_argument('--dropout', type=float, default=0, help='dropout in the model')
parser.add_argument('--gpu', type=int, default=4, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--arch_learning_rate', type=float, default=0.008, help='learning rate for arch encoding')
parser.add_argument('--arch_learning_rate_min', type=float, default=0.0005, help='min arch learning rate')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

parser.add_argument('--sample_num', type=int, default=5, help='sample numbers of the supernet')
parser.add_argument('--hidden_size',  type=int, default=64, help='default hidden_size in supernet')
parser.add_argument('--BN', action='store_true', default=False, help='Batch norm')
parser.add_argument('--LN', action='store_true', default=False, help='Layer norm')


#search space
parser.add_argument('--alpha_step', type=int, default=1, help='alpha update step comparing with w.')
parser.add_argument('--num_blocks', type=int, default=4, help='framework layers')
parser.add_argument('--num_cells', type=int, default=1, help='num of cells')
parser.add_argument('--cell_mode', type=str, default='full', choices=['repeat', 'diverse', 'full'])
parser.add_argument('--agg', type=str, default='sage', help='aggregations used in this framework')
parser.add_argument('--search_agg', type=bool, default=False, help='search aggregators')

#search algo
parser.add_argument('--algo', type=str, default='snas', help='search algorithm', choices=['darts', 'snas','random','bayes'])
parser.add_argument('--alpha_mode', type=str, default='valid', help='update alpha, with train/valid data', choices=['train', 'valid'])
parser.add_argument('--loc_mean', type=float, default=10.0, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
parser.add_argument('--temp', type=float, default=0.5, help='temp in softmax')
parser.add_argument('--temp_min', type=float, default=0.005, help='min temp in softmax')
parser.add_argument('--cos_temp', action='store_true', default=False, help='temp decay')
parser.add_argument('--w_update_epoch', type=int, default=1, help='epoches in update W')

parser.add_argument('--data_fold', type=int, default=10, help='x_fold cross-validation')



args = parser.parse_args()

def main(log_filename):
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    print('*************log_filename=%s************' % log_filename)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args.__dict__)


    data, num_nodes, num_features, num_classes = load_data(args.data, batch_size=args.batch_size, folds=args.data_fold)
    hidden_size = args.hidden_size
    # data = data.to(device)

    if 'ogb' in args.data:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model = Network(criterion, num_features, num_classes, hidden_size, dropout=args.dropout, args=args)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    model_optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay)

    arch_optimizer = torch.optim.Adam(
        model.arch_parameters(),
        lr=args.arch_learning_rate,
        weight_decay=args.arch_weight_decay)
    arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(arch_optimizer, float(args.epochs), eta_min=args.arch_learning_rate_min)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    temp_scheduler = utils.Temp_Scheduler(args.epochs, args.temp, args.temp, temp_min=args.temp_min)

    search_cost = 0
    # global epoch

    for epoch in range(args.epochs):
        t1 = time.time()
        lr = model_scheduler.get_last_lr()[0]
        if args.cos_temp:
            model.temp = temp_scheduler.step()
        else:
            model.temp = args.temp
        train_acc, train_loss = train(data, model, criterion, model_optimizer, arch_optimizer)
        model_scheduler.step()
        arch_scheduler.step()
        t2 = time.time()
        search_cost += (t2 - t1)

        # if (epoch + 1) % 10 == 0:
        valid_loss, valid_acc = infer(data, model, criterion, test=False)
        test_loss, test_acc = infer(data, model, criterion, test=True)
        print(
            'epoch={}, train_loss={}, train_acc={:.04f}, val_loss={}, valid_acc={:.04f}, test_loss:{},test_acc={:.04f}'.format(
                epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))


        if epoch % 1 == 0:
            genotype = model.genotype()
            # max_length = cal_range(genotype, args.num_blocks, args.num_cells, args.cell_mode)
            max_length=0
            logging.info('epoch: %d, lr: %e, temp: %e, max_length:%d', epoch, lr, model.temp, max_length)
            logging.info('genotype = %s, max_length=%s', genotype, max_length)

    logging.info('The search process costs %.2fs', search_cost)
    return genotype, valid_acc, test_acc


def train(data, model, criterion, model_optimizer, arch_optimizer):
    model.train()
    total_loss = 0
    accuracy = 0
    y_true = []
    y_pred = []

    #output, loss, accuracy
    train_iters = data[4].__len__()//args.w_update_epoch
    num_training_graphs = len(data[4].dataset)
    # print('num_graphs: {}/{}/{}'.format(data[4].dataset, data[5].dataset, data[6].dataset))
    if data[4].__len__() % args.w_update_epoch != 0:
        train_iters += 1
        num_training_graphs += len(data[4][0].dataset)
    # print(num_training_graphs)
    # print('train_iters:{}, train_data_num:{}'.format(train_iters, range(train_iters * args.w_update_epoch)))
    from itertools import cycle
    zip_train_data = list(zip(range(train_iters * args.w_update_epoch), cycle(data[4])))
    zip_valid_data = list(zip(range(train_iters), cycle(data[5])))

    for iter in range(train_iters):

        # update w
        for i in range(args.w_update_epoch):
            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()
            train_data = zip_train_data[iter*args.w_update_epoch+i][1].to(device)
            logits = model(train_data)
            accuracy += logits.max(1)[1].eq(train_data.y.view(-1)).sum().item()

            if 'ogb' in args.data:
                error_loss = criterion(logits.to(torch.float32), train_data.y.to(torch.float32))
                y_true.append(train_data.y.view(logits.shape).detach().cpu())
                y_pred.append(logits.detach().cpu())
            else:
                error_loss = criterion(logits, train_data.y.view(-1))

            total_loss += error_loss.item()
            arch_optimizer.zero_grad()
            # error_loss.backward(retain_graph=True)
            error_loss.backward()
            model_optimizer.step()

        #update alpha
        model_optimizer.zero_grad()
        if args.alpha_mode =='train':
            arch_optimizer.step()
        if args.alpha_mode =='valid':
            valid_data = zip_valid_data[iter][1].to(device)
            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()

            logits = model(valid_data)

            if 'ogb' in args.data:
                error_loss = criterion(logits.to(torch.float32), valid_data.y.to(torch.float32))
            else:
                error_loss = criterion(logits, valid_data.y.view(-1))

            error_loss.backward()
            arch_optimizer.step()
    if 'ogb' in args.data:
        evaluator = Evaluator(args.data)
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        return evaluator.eval(input_dict)['rocauc'], total_loss / num_training_graphs
    else:
        return accuracy/num_training_graphs, total_loss / num_training_graphs


def infer(data_, model, criterion, test=False, single_path=False):
    model.eval()
    total_loss = 0
    valid_acc, test_acc = 0, 0
    y_true, y_pred = [], []

    if test:
        data = data_[6]
    else:
        data = data_[5]

    for tmp_data in data:
        tmp_data = tmp_data.to(device)
        with torch.no_grad():
            logits = model(tmp_data, single_path=single_path)
            logits = logits.to(device)

        if 'ogb' in args.data:
            loss = criterion(logits.to(torch.float32), tmp_data.y.to(torch.float32))
            y_true.append(tmp_data.y.view(logits.shape).detach().cpu())
            y_pred.append(logits.detach().cpu())
        else:
            loss = criterion(logits, tmp_data.y)
        total_loss += loss.item()
        valid_acc += logits.max(1)[1].eq(tmp_data.y.view(-1)).sum().item()

    if 'ogb' in args.data:
        evaluator = Evaluator(args.data)
        y_pred = torch.cat(y_pred, dim=0).numpy()
        y_true = torch.cat(y_true, dim=0).numpy()
        acc_dict = {"y_true": y_true, "y_pred": y_pred}
        return total_loss/len(data.dataset), evaluator.eval(acc_dict)['rocauc']
    else:
        return total_loss/len(data.dataset), valid_acc/len(data.dataset)


def run_by_seed():
    res = []
    print('searched archs for {}...'.format(args.data))
    args.save = '{}-{}'.format(args.data, time.strftime("%Y%m%d-%H%M%S"))
    args.save = 'logs/search-{}'.format(args.save)
    if not os.path.exists(args.save):
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_filename = os.path.join(args.save, 'log.txt')
    if not os.path.exists(log_filename):
        init_logger('', log_filename, logging.INFO, False)

    for i in range(args.sample_num):
        seed = np.random.randint(0, 10000)
        args.seed = seed
        genotype, val_acc, test_acc = main(log_filename)
        res.append('seed={},genotype={},saved_dir={},val_acc={},test_acc={}'.format(seed, genotype, args.save, val_acc, test_acc))
    filename = 'exp_res/%s-searched-%s.txt' % (args.data, time.strftime('%Y%m%d-%H%M%S'),)
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('searched res for {} saved in {}'.format(args.data, filename))

if __name__ == '__main__':
    run_by_seed()
