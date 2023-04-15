import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import NA_PRIMITIVES,  SC_PRIMITIVES, FF_PRIMITIVES, READOUT_PRIMITIVES
from torch_geometric.nn import LayerNorm, BatchNorm

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")

class NaOp(nn.Module):

    def __init__(self, in_dim, out_dim,op_name):
        super(NaOp, self).__init__()
        self.op = NA_OPS[op_name](in_dim, out_dim)

    def reset_parameters(self):
        self.op.reset_parameters()

    def forward(self, x, edge_index):
        return self.op(x, edge_index)

class NAMixedOp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NAMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in NA_PRIMITIVES:
            op = NA_OPS[primitive](in_dim, out_dim)
            self._ops.append(op)

    def reset_parameters(self):
        for op in self._ops:
            op.reset_parameters()

    def forward(self, x, edge_index, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(x, edge_index))
        return sum(mixed_res)

class ScMixedOp(nn.Module):
    def __init__(self):
        super(ScMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in SC_PRIMITIVES:
            op = SC_OPS[primitive]()
            self._ops.append(op)

    def forward(self, x, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(x))
        return sum(mixed_res)

class LaMixedOp(nn.Module):

    def __init__(self, hidden_size, num_layers=None):
        super(LaMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in FF_PRIMITIVES:
            op = FF_OPS[primitive](hidden_size, num_layers)
            self._ops.append(op)

    def reset_parameters(self):
        for op in self._ops:
            op.reset_parameters()

    def forward(self, x, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * F.relu(op(x)))
        return sum(mixed_res)

class ReadoutMixedOp(nn.Module):
    def __init__(self, hidden):
        super(ReadoutMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in READOUT_PRIMITIVES:
            op = READOUT_OPS[primitive](hidden)
            self._ops.append(op)

    def reset_parameters(self):
        for op in self._ops:
            op.reset_parameters()

    def forward(self, x, batch, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            tmp_res = w * op(x, batch)
            # print('readout', tmp_res.size())
            mixed_res.append(tmp_res)
        return sum(mixed_res)

def process_feature(features, size):
    new_feature = []
    for feature in features:
        new_feature += [feature[:size]]
    return new_feature

class Network(nn.Module):

    def __init__(self, criterion, in_dim, out_dim, hidden_size, dropout=0.5, args=None, evaluate=False):
        super(Network, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_blocks = args.num_blocks
        self.num_cells = args.num_cells
        self.cell_mode = args.cell_mode
        self._criterion = criterion
        self.dropout = dropout
        self.args = args
        self.evaluate = evaluate
        self.temp = args.temp

        #pre-process Node 0
        self.lin1 = nn.Linear(in_dim, hidden_size)

        #node aggregator op, intermediate nodes
        self.gnn_layers = nn.ModuleList()
        if self.cell_mode == 'repeat':
            num_searched_agg = int(self.num_blocks / self.num_cells)
        else:
            num_searched_agg = self.num_blocks
        self.num_searched_agg = num_searched_agg
        if self.args.search_agg: #search agg.
            for i in range(num_searched_agg):
                self.gnn_layers.append(NAMixedOp(hidden_size, hidden_size))
        else: #fixed agg
            aggs = [self.args.agg] * self.num_blocks
            for i in range(self.num_blocks):
                self.gnn_layers.append(NaOp(hidden_size, hidden_size, op_name=aggs[i]))

        #skip op
        num_node_per_cell = int(self.num_blocks / self. num_cells)
        self.num_node_per_cell = num_node_per_cell

        if self.cell_mode == 'full':
            num_searched_skip = (self.args.num_blocks + 2) * (self.args.num_blocks + 1) / 2
        elif self.cell_mode == 'repeat':
            num_searched_skip = (num_node_per_cell + 2) * (num_node_per_cell + 1) / 2
        else: # diverse
            num_searched_skip = self.num_cells * (num_node_per_cell + 2) * (num_node_per_cell + 1) / 2

        self.num_edges = int(num_searched_skip)
        self.skip_op = nn.ModuleList()
        for i in range(self.num_edges):
            self.skip_op.append(ScMixedOp())

        # fuse function in each layer.
        self.fuse_funcs = nn.ModuleList()
        if self.cell_mode == 'full':
            for i in range(self.num_blocks + self.num_cells):
                self.fuse_funcs.append(LaMixedOp(hidden_size, i + 1))
            num_searched_ff = self.num_blocks + self.num_cells
        elif self.cell_mode == 'repeat':
            for node in range(num_node_per_cell + 1):
                self.fuse_funcs.append(LaMixedOp(hidden_size, node + 1))
            num_searched_ff = num_node_per_cell + 1
        elif self.cell_mode == 'diverse':
            for cell in range(self.num_cells):
                for node in range(num_node_per_cell + 1):
                    self.fuse_funcs.append(LaMixedOp(hidden_size, node + 1))
            num_searched_ff = self.num_blocks + self.num_cells
        self.num_searched_ff = num_searched_ff

        self.cell_output_lins = nn.ModuleList()
        for i in range(self.num_cells):
            self.cell_output_lins.append(Linear(hidden_size, hidden_size))

        self.readout_layers = ReadoutMixedOp(hidden_size)
        self.readout_lin = Linear(hidden_size, hidden_size)
        self.classifier = Linear(hidden_size, out_dim)


        #extra ops
        self.lns = nn.ModuleList()
        for i in range(self.num_blocks):
            self.lns.append(LayerNorm(hidden_size, affine=False))

        self.bns = nn.ModuleList()
        for i in range(self.num_blocks):
            self.bns.append(BatchNorm(hidden_size, affine=False))

        self._initialize_alphas()

        self.reset_parameters()
    def reset_parameters(self):
        self.lin1.reset_parameters()
        for agg in self.gnn_layers:
            agg.reset_parameters()
        for ff in self.fuse_funcs:
            ff.reset_parameters()
        for lin in self.cell_output_lins:
            lin.reset_parameters()
        self.readout_layers.reset_parameters()
        self.readout_lin.reset_parameters()
        self.classifier.reset_parameters()
        
            
    def _get_categ_mask(self, alpha):
        log_alpha = alpha
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self.temp)
        return one_hot
    def _get_softmax_temp(self, alpha):
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax(alpha / self.temp)
        return one_hot

    def get_one_hot_alpha(self, alpha):
        one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
        idx = torch.argmax(alpha, dim=-1)
        for i in range(one_hot_alpha.size(0)):
            one_hot_alpha[i, idx[i]] = 1.0
        return one_hot_alpha

    def forward(self, data, single_path=False):

        if self.training:
            if self.args.algo == 'darts':
                self.sc_weights = self._get_softmax_temp(self.sc_alphas)
                self.ff_weights = self._get_softmax_temp(self.ff_alphas)
                self.readout_weights = self._get_softmax_temp(self.readout_alphas)

                if self.args.search_agg:
                    self.agg_weights = self._get_softmax_temp(self.agg_alphas)

            elif self.args.algo == 'snas':
                self.sc_weights = self._get_categ_mask(self.sc_alphas)
                self.ff_weights = self._get_categ_mask(self.ff_alphas)
                self.readout_weights = self._get_categ_mask(self.readout_alphas)

                if self.args.search_agg:
                    self.agg_weights = self._get_categ_mask(self.agg_alphas)
        else:
            if single_path:
                self.sc_weights = self.get_one_hot_alpha(self.sc_alphas)
                self.ff_weights = self.get_one_hot_alpha(self.ff_alphas)
                if self.args.search_agg:
                    self.agg_weights = self.get_one_hot_alpha(self.agg_alphas)

        output = self.forward_model(data)
        return output

    def _get_edge_id(self, cell, cur_node, input_node):
            if self.cell_mode =='full':
                edge_id = (cur_node + 1) * cur_node / 2 + input_node
            elif self.cell_mode == 'diverse':
                num_edges_per_cell = (self.num_node_per_cell + 2) * (self.num_node_per_cell + 1) / 2
                edge_id = cell * num_edges_per_cell + int((cur_node + 1) * cur_node / 2) + input_node
            else: #'repeat'
                edge_id = (cur_node + 1) * cur_node / 2 + input_node
            return int(edge_id)

    def _get_ff_id(self, cell, cur_node):

        if self.cell_mode == 'repeat':
            return cur_node
        else: #diverse or full
            return cell * (self.num_node_per_cell + 1) + cur_node


    def forward_model(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        features = []

        # input node 0
        cell_output = []
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        features += [x]
        cell_output += [x]
        
        # print('num_edges:{}, num_ff:{}'.format(self.num_edges, self.num_searched_ff))
        num_node_per_cell = int(self.num_blocks / self.num_cells)

        for cell in range(self.num_cells):
            for node in range(num_node_per_cell + 1):
                # select inputs
                layer_input = []
                for i in range(node + 1):
                    edge_id = self._get_edge_id(cell, node, i)
                    layer_input += [self.skip_op[edge_id](features[i], self.sc_weights[edge_id])]
                    # print('selection: {},{},{},{}'.format(cell, node, i, edge_id))

                # fuse features
                ff_id = self._get_ff_id(cell, node)
                tmp_input = self.fuse_funcs[ff_id](layer_input, self.ff_weights[ff_id])

                # aggregation
                agg_id = cell * self.num_node_per_cell + node

                if node == self.num_node_per_cell:
                    x = self.cell_output_lins[cell](tmp_input)
                elif self.args.search_agg:
                    x = self.gnn_layers[agg_id](tmp_input, edge_index, self.agg_weights[agg_id])
                else:
                    x = self.gnn_layers[agg_id](tmp_input, edge_index)

                x = F.relu(x)
                if node != self.num_node_per_cell: #for the aggregation results
                    if self.args.BN:
                        x = self.bns[agg_id](x)
                    elif self.args.LN:
                        x = self.lns[agg_id](x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # output
                features += [x]

            # reset the input for each cell.
            features = [x]
            cell_output += [x]

             
        output = self.readout_layers(x, batch, self.readout_weights[0])
        output = F.relu(self.readout_lin(output))
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.classifier(output)
        return output


    def _initialize_alphas(self):
        num_sc_ops = len(SC_PRIMITIVES)
        num_ff_ops = len(FF_PRIMITIVES)
        num_na_ops = len(NA_PRIMITIVES)
        num_readout_ops = len(READOUT_PRIMITIVES)
        if self.args.algo in ['darts', 'random', 'bayes']:
            self.sc_alphas = Variable(1e-3 * torch.randn(self.num_edges, num_sc_ops).cuda(), requires_grad=True)
            self.ff_alphas = Variable(1e-3 * torch.randn(self.num_searched_ff, num_ff_ops).cuda(), requires_grad=True)
            self.readout_alphas = Variable(1e-3 * torch.randn(1, num_readout_ops).cuda(), requires_grad=True)
            if self.args.search_agg:
                self.agg_alphas = Variable(1e-3 * torch.randn(self.num_blocks, num_na_ops).cuda(), requires_grad=True)

        elif self.args.algo == 'snas':
            self.sc_alphas = Variable(torch.ones(self.num_edges, num_sc_ops).normal_(self.args.loc_mean, self.args.loc_std).cuda(), requires_grad=True)
            self.ff_alphas = Variable(torch.ones(self.num_searched_ff, num_ff_ops).normal_(self.args.loc_mean, self.args.loc_std).cuda(), requires_grad=True)
            self.readout_alphas = Variable(torch.ones(1, num_readout_ops).normal_(self.args.loc_mean, self.args.loc_std).cuda(), requires_grad=True)

            if self.args.search_agg:
                self.agg_alphas = Variable(torch.ones(self.num_blocks, num_na_ops).normal_(self.args.loc_mean, self.args.loc_std).cuda(),requires_grad=True)

        if self.args.search_agg:
            self._arch_parameters = [
                self.sc_alphas,
                self.ff_alphas,
                self.agg_alphas,
                self.readout_alphas
            ]
        else:
            self._arch_parameters = [
                self.sc_alphas,
                self.ff_alphas,
                self.readout_alphas
            ]

    def arch_parameters(self):
        return self._arch_parameters

    def _parse(self, sc_weights, la_weights):
        gene = []

        if '||' in self.args.agg:
            aggs = self.args.agg.split('||')
            gene.append(aggs[:])
        else:
            aggs = [self.args.agg] * self.args.num_layers
        gene += aggs

        sc_indices = torch.argmax(sc_weights, dim=-1)
        for k in sc_indices:
            gene.append(SC_PRIMITIVES[k])
        la_indices = torch.argmax(la_weights, dim=-1)
        for k in la_indices:
            gene.append(FF_PRIMITIVES[k])
        return '||'.join(gene)
    def sparse_single(self,weights, opsets):
        gene = []
        indices = torch.argmax(weights, dim=-1)
        for k in indices:
            gene.append(opsets[k])
        return gene

    def genotype(self, sample=False):

        gene = []

        # agg
        if self.args.search_agg:
            gene += self.sparse_single(F.softmax(self.agg_alphas, dim=-1).data.cpu(), NA_PRIMITIVES)
        else:
            gene += [self.args.agg] * self.num_blocks

        # topology
        if self.cell_mode == 'repeat':
            for cell in range(self.num_cells):
                gene += self.sparse_single(F.softmax(self.sc_alphas, dim=-1).data.cpu(), SC_PRIMITIVES)
            for cell in range(self.num_cells):
                gene += self.sparse_single(F.softmax(self.ff_alphas, dim=-1).data.cpu(), FF_PRIMITIVES)
        else:
            gene += self.sparse_single(F.softmax(self.sc_alphas, dim=-1).data.cpu(), SC_PRIMITIVES)
            gene += self.sparse_single(F.softmax(self.ff_alphas, dim=-1).data.cpu(), FF_PRIMITIVES)
        gene += self.sparse_single(F.softmax(self.readout_alphas, dim=-1).data.cpu(), READOUT_PRIMITIVES)

        return '||'.join(gene)

