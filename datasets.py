from torch import cat
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree, to_dense_adj, is_undirected, to_dense_adj
import torch_geometric.transforms as T
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

path = './data/'


class HandleNodeAttention(object):
    def __call__(self, data):
        data.attn = torch.softmax(data.x[:, 0], dim=0)
        data.x = data.x[:, 1:]
        return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.uint8, device=index.device)
    mask[index] = 1
    return mask


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0])
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def load_tudata(dataset_name='DD', cleaned=False, split_seed=12345, batch_size=32, remove_large_graph=True, folds=10):
    dataset = TUDataset(path, dataset_name, cleaned=cleaned)
    dataset.data.edge_attr = None

    # load and process
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0])]
            max_degree = max(max_degree, degs[-1].max().item())
            max_degree = int(max_degree)
        print('max degree:', max_degree)
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    num_nodes = max_num_nodes = 0
    skf = StratifiedKFold(folds, shuffle=True, random_state=split_seed)
    idx = [torch.from_numpy(i) for _, i in skf.split(torch.zeros(len(dataset)), dataset.data.y[:len(dataset)])]
    print('{} fold split'.format(folds))
    if folds == 10:
        split = [cat(idx[:8], 0), cat(idx[8:9], 0), cat(idx[9:], 0)]
    elif folds == 5:
        split = [cat(idx[:3], 0), cat(idx[3:4], 0), cat(idx[4:], 0)]
    else:
        print('error split')
        
    train_dataset = dataset[split[0]]
    val_dataset = dataset[split[1]]
    test_dataset = dataset[split[2]]
    print('train:{}, val:{}, test:{}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    print('num feature:', num_features, num_classes)

    return [dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader], \
           num_nodes, num_features, num_classes


def load_textdata(dataset_name='DD', cleaned=False, split_seed=12345, batch_size=32, remove_large_graph=True):
    dataset = torch.load(path + '{}.pt'.format(dataset_name))
    num_nodes = max_num_nodes = 0
    for data in dataset:
        num_nodes += data.x.shape[0]
        max_num_nodes = max(data.x.shape[0], max_num_nodes)

    skf = StratifiedKFold(10, shuffle=True, random_state=split_seed)
    y = [data.y.item() for data in dataset]
    idx = [torch.from_numpy(i) for _, i in skf.split(torch.zeros(len(dataset)), y)]
    split = [cat(idx[:8], 0), cat(idx[8:9], 0), cat(idx[9:], 0)]


    # torch.save(split, 'mr_split.pt')
    # for i in range(10):
    #     data = dataset[i]
    #     print(data.x.shape, data.edge_index.shape, data.edge_index.max())
    # dataset[14].x.shape, dataset[14].edge_index.shape, dataset[14].edge_index.max()
    # print(split[0][:10])
    # print(split[1][:10])
    # print(split[2][:10])
    train_dataset = [dataset[i] for i in split[0]]
    test_dataset = [dataset[i] for i in split[1]]
    val_dataset = [dataset[i] for i in split[2]]

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)



    num_classes = max(y) + 1
    num_features = dataset[0].x.shape[1]

    return [dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader], \
           num_nodes, num_features, num_classes


def load_data(dataset_name='DD', cleaned=False, split_seed=12345, batch_size=32, remove_large_graph=True, folds=10):
    if dataset_name in ['mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2']:
        return load_textdata(dataset_name, cleaned, split_seed, batch_size, remove_large_graph)
    elif 'ogb' in dataset_name:
        dataset = PygGraphPropPredDataset(name=dataset_name, root=path)
        print('using simple feature')
        dataset.data.x = (dataset.data.x[:, :2]).type(torch.FloatTensor)
        dataset.data.edge_attr = None

        num_features = dataset.data.x.size(1)
        print('dataset num_tasks:', dataset.num_tasks)

        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

        return [dataset, None, None, None, train_loader, val_loader, test_loader], 0, num_features, dataset.num_tasks

    else:
        return load_tudata(dataset_name, cleaned, split_seed, batch_size, remove_large_graph, folds=folds)


def load_k_fold(dataset_name, dataset, folds, batch_size):
    if dataset_name in ['mr', 'ohsumed', 'R8', 'R52', 'TREC', 'ag_news', 'WebKB', 'SST1', 'SST2']:
        y = [data.y.item() for data in dataset]
    else:
        y = dataset.data.y[:len(dataset)]

    print('{} fold split'.format(folds))
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    data_10fold = []
    for i in range(folds):
        data_ith = [0, 0, 0, 0]  # align with 811 split process.
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0

        train_mask = train_mask.nonzero().view(-1)

        data_ith.append(DataLoader([dataset[i] for i in train_mask], batch_size, shuffle=True))
        data_ith.append(DataLoader([dataset[i] for i in val_indices[i]], batch_size, shuffle=True))
        data_ith.append(DataLoader([dataset[i] for i in test_indices[i]], batch_size, shuffle=True))
        data_10fold.append(data_ith)

    return data_10fold

from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from collections import Counter
import numpy as np
def cal_diameter(dataset):
    diameter = []
    data = dataset[0]
    for example in data:
        edge_index = example.edge_index
        N = example.x.size(0)
        if is_undirected(edge_index):
            adj = to_dense_adj(edge_index, max_num_nodes=N)[0]
            distance = torch.tensor(shortest_path(csr_matrix(adj), directed=False))
            max_length = distance[(1-torch.isinf(distance).float()).bool()].max().item()
            diameter.append(max_length)
        else:
            print('directed graph!')
    # diameter = np.array(diameter)
    # return Counter(diameter)
    diameter = torch.tensor(diameter)
    return torch.mean(diameter)



