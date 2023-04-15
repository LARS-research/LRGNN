import  torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def get_adj(edge_weights, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    start = 0
    for i in range(num_nodes - 1):
        input_edges = i + 1
        adj[:input_edges, i + 1] = edge_weights[start:(start + input_edges)]
        start = start + input_edges
    return adj
def subset(a, b, index, res):
    if index == len(a):
        res.append(b)
    else:
        c = b[:]
        b.append(a[index])
        subset(a, b, index + 1, res)
        subset(a, c, index + 1, res)

def get_paths_prob(c, paths, num_blocks, path_end=5):
    paths_prob=[]
    adj = get_adj(c, num_blocks+2)  #layers+input+output
    for path in paths:
        if len(path) == 0:
            prob = adj[0, path_end]
        else:
            prob = 1.0
            edge_start = 0
            for node in path:
                edge_end = node
                prob = prob * adj[edge_start, edge_end]
                edge_start = node
            prob = prob * adj[edge_start, path_end]
        paths_prob.append(prob)
    return np.array(paths_prob)

def cal_cell_matrix(num_blocks, edge_weights):
    range_p_matrix = np.zeros((num_blocks + 2, num_blocks + 2))
    for node in range(num_blocks + 1):
        res = []
        subset([i + 1 for i in range(node)], [], 0, res)
        # print(res, len(res))
        res_length = np.array([len(i) for i in res])
        paths_prob = get_paths_prob(edge_weights, res, num_blocks, path_end=node + 1)
        paths_prob = np.array(paths_prob)
        # print(paths_prob)

        range_p = []
        for i in range(node + 1):
            idx = np.array(np.where(res_length == i)[0])
            range_p.append(max(paths_prob[idx[:]]))
        range_p = np.array(range_p)
        range_p_matrix[node + 1, :node + 1] = range_p
    return range_p_matrix

def plot_range(arch, num_blocks, num_cells, cell_mode, file_name=None, performance=None):
    arch = arch.split('||')
    edge_weights = []
    for i in arch:
        if i == 'zero':
            edge_weights.append(0)
        else:
            edge_weights.append(1)
    edge_weights = torch.tensor(edge_weights)
    if cell_mode == 'full':
        # range_matrix = np.zeros((num_blocks + num_cells, num_blocks + num_cells))
        range_matrix = cal_cell_matrix(num_blocks, edge_weights)[1:, :-1]
        num_nodes_per_cell = num_blocks
    else:
        range_matrix = np.zeros((num_blocks + num_cells, num_blocks + num_cells))
        num_nodes_per_cell = int(num_blocks / num_cells)
        num_edges_per_cell = int((num_nodes_per_cell+2)*(num_nodes_per_cell+1) / 2)
        start_range = 0
        for cell in range(num_cells):
            # print('cell edges:',edge_weights[num_edges_per_cell * cell: num_edges_per_cell * (cell + 1)])
            tmp_matrix = cal_cell_matrix(num_nodes_per_cell, edge_weights[num_edges_per_cell * cell: num_edges_per_cell * (cell + 1)])[1:, :-1]
            # print('cell idx:', (num_nodes_per_cell + 1) * cell, (num_nodes_per_cell + 1) * (cell + 1))
            tmp_range = np.max(np.where(tmp_matrix[-1, :] == 1))
            range_matrix[(num_nodes_per_cell + 1) * cell: (num_nodes_per_cell + 1) * (cell + 1), start_range: start_range + num_nodes_per_cell+1] = \
                tmp_matrix
            start_range += tmp_range

    #longest_range
    final_range = np.where(range_matrix[-1, :] == 1.0)[0]
    if len(final_range)==0:
        return 0
    else:
        return final_range.max()



def cal_range(full_arch, num_blocks, num_cells, cell_mode):
    arch = full_arch.split('||')[num_blocks:-num_cells-num_blocks-1]
    return plot_range('||'.join(arch), num_blocks, num_cells, cell_mode)
