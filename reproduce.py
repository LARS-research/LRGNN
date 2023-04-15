from train4tune import main
import argparse

parser = argparse.ArgumentParser("pas")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--data', type=str, default='DD')
parser.add_argument('--ntimes', type=int, default=5)
args1 = parser.parse_args()


NCI1_B12C1_full_params = {'gpu': 2, 'data': 'NCI1', 'arch_filename': 'exp_res/NCI1-searched-20220620-202058.txt', 
               'arch': 'gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||identity||identity||zero||identity||zero||identity||identity||identity||zero||zero||identity||zero||identity||identity||identity||identity||zero||zero||zero||identity||identity||identity||zero||identity||zero||zero||identity||identity||zero||zero||zero||identity||zero||zero||zero||zero||zero||zero||identity||identity||identity||identity||zero||zero||identity||zero||identity||identity||identity||zero||zero||zero||identity||identity||identity||zero||zero||identity||identity||zero||zero||zero||zero||zero||identity||zero||identity||zero||zero||zero||identity||zero||zero||identity||identity||zero||identity||zero||identity||identity||zero||identity||identity||zero||identity||zero||identity||identity||zero||identity||zero||lstm||concat||mean||max||max||att||mean||concat||concat||att||att||mean||sum||global_sum', 
               'num_blocks': 12, 'num_cells': 1, 'cell_mode': 'full', 'agg': 'gcn', 'search_agg': False, 'model_type': 'NONE', 
               'hidden_size': 512, 'learning_rate': 0.008814121411828977, 'weight_decay': 0.0, 'ft_mode': '10fold', 'BN': True, 
               'LN': False, 'rml2': True, 'rmdropout': True, 'hyper_epoch': 20, 'epochs': 100, 'cos_lr': True, 'lr_min':0.0, 
               'std_times': 5, 'batch_size': 128, 'tune_id': -1, 'seed': 5609, 'dropout': 0.0, 
               'model': 'f2gnn', 'optimizer': 'adagrad', 'rnd_num': 1, 'grad_clip': 5, 'momentum': 0.9, 'data_fold':10}

NCI109_B8C1_full_params = {'gpu': 7, 'data': 'NCI109', 'arch_filename': '', 
                           'arch': 'gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||identity||zero||identity||zero||zero||identity||zero||identity||zero||identity||identity||zero||zero||zero||identity||identity||identity||zero||zero||identity||zero||identity||zero||identity||zero||zero||identity||zero||zero||zero||zero||identity||identity||identity||zero||zero||zero||zero||zero||identity||identity||identity||identity||zero||identity||concat||max||mean||lstm||att||att||lstm||lstm||sum||global_sum', 
                           'num_blocks': 8, 'num_cells': 1, 'cell_mode': 'full', 'ft_mode': '10fold', 'BN': True, 'LN':False, 'rml2': True, 
                           'rmdropout': True, 'hyper_epoch': 20, 'epochs': 100, 'cos_lr': True, 'lr_min': 0.0, 'std_times': 5, 'batch_size': 128,
                             'tune_id': -1, 'seed': 2, 'dropout': 0.0, 'hidden_size': 128, 'learning_rate': 0.01311902474829272, 'model': 'f2gnn', 
                             'optimizer': 'adagrad', 'weight_decay': 0.0, 'rnd_num': 1, 'grad_clip': 5, 'momentum': 0.9, 'data_fold':10}

DD_B12C1_full_params = {'gpu': 4, 'data': 'DD', 'arch_filename': '', 
                        'arch': 'gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||identity||identity||zero||zero||identity||zero||zero||zero||zero||identity||zero||identity||zero||identity||zero||zero||identity||identity||identity||identity||identity||identity||identity||identity||zero||zero||zero||identity||zero||zero||zero||zero||zero||zero||zero||zero||identity||zero||identity||identity||identity||zero||zero||identity||zero||identity||zero||zero||zero||zero||identity||identity||zero||zero||identity||zero||zero||identity||zero||zero||zero||zero||identity||identity||zero||identity||zero||identity||zero||zero||zero||identity||zero||identity||zero||identity||identity||identity||identity||identity||identity||identity||identity||identity||identity||identity||identity||identity||identity||identity||identity||max||concat||sum||att||att||max||lstm||concat||concat||max||max||mean||max||global_sum', 
                        'num_blocks': 12, 'num_cells': 1, 'cell_mode': 'full', 'agg': 'gcn', 'search_agg': False, 'model_type': 'NONE',
                          'hidden_size': 128, 'learning_rate': 0.003313481992166702, 'weight_decay': 0.0002999743602588061, 
                          'ft_mode': '10fold', 'BN': True, 'LN': False, 'rml2': False, 'rmdropout': False, 'hyper_epoch': 20, 
                          'epochs': 100, 'cos_lr': True, 'lr_min': 0.0, 'std_times': 5, 'batch_size': 32, 'tune_id': 4, 
                          'seed': 6422, 'dropout': 0.2, 'model': 'f2gnn', 'optimizer': 'adam', 'rnd_num': 1, 'grad_clip': 5, 'momentum': 0.9, 'data_fold':10}
PROTEINS_B8C1_full_params = {'gpu': 2, 'data': 'PROTEINS', 'arch_filename': '', 
                             'arch': 'gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||identity||zero||identity||zero||zero||zero||zero||zero||identity||zero||zero||zero||zero||identity||identity||identity||identity||zero||identity||identity||identity||identity||zero||identity||zero||identity||zero||identity||zero||zero||identity||zero||identity||zero||zero||zero||identity||identity||identity||identity||identity||zero||zero||identity||zero||mean||max||lstm||lstm||max||mean||concat||lstm||att||global_sum', 
                             'num_blocks': 8, 'num_cells': 1, 'cell_mode': 'full', 'ft_mode': '10fold', 'BN': True, 'LN': False, 'rml2': False, 
                             'rmdropout': False, 'hyper_epoch': 20, 'epochs': 100, 'cos_lr': True, 'lr_min': 0.0, 'std_times': 5, 
                             'batch_size': 128, 'tune_id': -1, 'seed': 2, 'dropout': 0.2, 'hidden_size': 256, 'learning_rate': 0.0038948647662910996, 
                             'model': 'f2gnn', 'optimizer': 'adam', 'weight_decay': 0.0006578269294157753, 'rnd_num': 1, 'grad_clip': 5, 'momentum': 0.9, 'data_fold':10}
IMDBB_B8C1_full_params = {'gpu': 5, 'data': 'IMDB-BINARY', 'arch_filename': '', 
                          'arch': 'gcn||gcn||gcn||gcn||gcn||gcn||gcn||gcn||identity||identity||zero||zero||identity||identity||identity||identity||zero||identity||zero||identity||identity||identity||zero||zero||zero||zero||identity||identity||zero||identity||zero||identity||identity||zero||identity||identity||zero||identity||zero||identity||identity||identity||zero||identity||identity||zero||identity||identity||zero||zero||identity||zero||zero||concat||sum||max||concat||concat||max||lstm||att||att||global_sum', 
                          'num_blocks': 8, 'num_cells': 1, 'cell_mode': 'full', 'ft_mode': '10fold', 'BN': True, 'LN': False, 'rml2': False, 
                          'rmdropout': False, 'hyper_epoch': 20, 'epochs': 100, 'cos_lr': True, 'lr_min': 0.0, 'std_times': 5, 'batch_size': 128,
                          'tune_id': -1, 'seed': 2, 'dropout': 0.1, 'hidden_size': 256, 'learning_rate': 0.005636433255231337, 
                          'model': 'f2gnn', 'optimizer': 'adagrad', 'weight_decay': 0.0007630013275383828, 'rnd_num': 1, 'grad_clip': 5, 
                          'momentum': 0.9, 'data_fold':10}

notes = {
'NCI1_B12C1_full': '82.51(1.37)',
'NCI109_B8C1_full': '81.39(1.92)',
'DD_B12C1_full': '78.18(2.02)',
'PROTEINS_B8C1_full':'75.39(4.40)',
'IMDBB_B8C1_full': '76.20(5.18)'
}

params_dict={
    'DD': 'DD_B12C1_full_params',
    'NCI1': 'NCI1_B12C1_full_params',
    'NCI109': 'NCI109_B8C1_full_params',
    'PROTEINS': 'PROTEINS_B8C1_full_params',
    'IMDB-BINARY': 'IMDBB_B8C1_full_params'
}
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

args = Dict()
param_name = params_dict[args1.data]
params = locals()[param_name]
print(params)

params['gpu'] = args1.gpu
for k, v in params.items():
    args[k] = v

print('*'*35, 'reproduce {}'.format(args1.data), '*'*35)
print(notes[param_name[:-7]])


for i in range(args1.ntimes):
    valid_acc, test_acc, test_std, args = main(args)
    print('{}/{}: valid_acc:{:.04f}, test_acc: {:.04f}+-{:.04f}'.format(i+1, args1.ntimes,  valid_acc, test_acc, test_std))


