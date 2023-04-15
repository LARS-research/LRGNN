import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, LSTM, GRU

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, JumpingKnowledge,SAGPooling
from torch_geometric.nn import GINConv
from torch.nn import Conv1d
from pyg_gnn_layer import GeoLayer
# from gin_conv import GINConv2
# from gcn_conv import GCNConv2
# from geniepath import GeniePathLayer
from torch_geometric.nn import global_add_pool,global_mean_pool,global_max_pool,global_sort_pool,GlobalAttention,Set2Set


NA_OPS = {
    'sage': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sage'),
    'sage_sum': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'sum'),
    'sage_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'max'),
    'gcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn'),
    'gat': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat'),
    'gin': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gin'),
    'gat_sym': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat_sym'),
    'gat_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'linear'),
    'gat_cos': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'cos'),
    'gat_generalized_linear': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'generalized_linear'),
    # 'geniepath': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'geniepath'),
}



SC_OPS={
    'zero': lambda: Zero(),
    'identity': lambda: Identity(),
}

FF_OPS = {
    'sum': lambda hidden_size, num_layers: LaAggregator('sum', hidden_size, num_layers),
    'mean': lambda hidden_size, num_layers: LaAggregator('mean', hidden_size, num_layers),
    'max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
    'concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
    'lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers),
    'gate': lambda hidden_size, num_layers: LaAggregator('gate', hidden_size, num_layers),
    'att': lambda hidden_size, num_layers: LaAggregator('att', hidden_size, num_layers),
}

READOUT_OPS = {
    "global_mean": lambda hidden: Readout_func('mean', hidden),
    "global_sum": lambda hidden: Readout_func('add', hidden),
    "global_max": lambda hidden: Readout_func('max', hidden),
    'mean_max': lambda hidden: Readout_func('mema', hidden),
    "none": lambda hidden: Readout_func('none', hidden),
    'global_att': lambda hidden: Readout_func('att', hidden),
    'global_sort': lambda hidden: Readout_func('sort', hidden),
    'set2set': lambda hidden: Readout_func('set2set', hidden)
}

class NaAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggregator):
        super(NaAggregator, self).__init__()
        #aggregator, K = agg_str.split('_')
        if 'sage' == aggregator:
            self._op = SAGEConv(in_dim, out_dim, normalize=True)
        if 'gcn' == aggregator:
            self._op = GCNConv(in_dim, out_dim)
        if 'gat' == aggregator:
            heads = 4
            out_dim /= heads
            self._op = GATConv(in_dim, int(out_dim), heads=heads, dropout=0.5)
        if 'gin' == aggregator:
            nn1 = Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
            self._op = GINConv(nn1)
        if aggregator in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
            heads = 8
            out_dim /= heads
            self._op = GeoLayer(in_dim, int(out_dim), heads=heads, att_type=aggregator, dropout=0.5)
        if aggregator in ['sum', 'max']:
            self._op = GeoLayer(in_dim, out_dim, att_type='const', agg_type=aggregator, dropout=0.5)
        # if aggregator in ['geniepath']:
        #     self._op = GeniePathLayer(in_dim, out_dim)

    def reset_parameters(self):
        self._op.reset_parameters()

    def forward(self, x, edge_index):
        return self._op(x, edge_index)



class LaAggregator(nn.Module):

    def __init__(self, mode, hidden_size, num_layers=3):
        super(LaAggregator, self).__init__()
        self.mode = mode
        if mode in ['lstm', 'cat', 'max']:
            self.jump = JumpingKnowledge(mode, hidden_size, num_layers=num_layers)
        elif mode == 'att':
            self.att = Linear(hidden_size, 1)

        if mode == 'cat':
            self.lin = Linear(hidden_size * num_layers, hidden_size)
        else:
            self.lin = Linear(hidden_size, hidden_size)
    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.mode in ['lstm', 'cat', 'max']:
            self.jump.reset_parameters()
        if self.mode == 'att':
            self.att.reset_parameters()

    def forward(self, xs):
        if self.mode in ['lstm', 'cat', 'max']:
            output = self.jump(xs)
        elif self.mode == 'sum':
            output = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            output = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'att':
            input = torch.stack(xs, dim=-1).transpose(1, 2)
            weight = self.att(input)
            weight = F.softmax(weight, dim=1)# cal the weightes of each layers and each node
            output = torch.mul(input, weight).transpose(1, 2).sum(dim=-1) #weighte sum

        return self.lin(F.relu(output))

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)


class Readout_func(nn.Module):
    def __init__(self, readout_op, hidden):

        super(Readout_func, self).__init__()
        self.readout_op = readout_op

        if readout_op == 'mean':
            self.readout = global_mean_pool

        elif readout_op == 'max':
            self.readout = global_max_pool

        elif readout_op == 'add':
            self.readout = global_add_pool

        elif readout_op == 'att':
            self.readout = GlobalAttention(Linear(hidden, 1))

        elif readout_op == 'set2set':
            processing_steps = 2
            self.readout = Set2Set(hidden, processing_steps=processing_steps)
            self.s2s_lin = Linear(hidden*processing_steps, hidden)


        elif readout_op == 'sort':
            self.readout = global_sort_pool
            self.k = 10
            self.sort_conv = Conv1d(hidden, hidden, 5)#kernel size 3, output size: hidden,
            self.sort_lin = Linear(hidden*(self.k-5 + 1), hidden)
        elif readout_op =='mema':
            self.readout = global_mean_pool
            self.lin = Linear(hidden*2, hidden)
        elif readout_op == 'none':
            self.readout = global_mean_pool
        # elif self.readout_op == 'mlp':

    def reset_parameters(self):
        if self.readout_op =='sort':
            self.sort_conv.reset_parameters()
            self.sort_lin.reset_parameters()
        if self.readout_op in ['set2set', 'att']:
            self.readout.reset_parameters()
        if self.readout_op =='set2set':
            self.s2s_lin.reset_parameters()
        if self.readout_op == 'mema':
            self.lin.reset_parameters()
    def forward(self, x, batch):
        #sparse data
        if self.readout_op == 'none':
            x = self.readout(x, batch)
            return x.mul(0.)
            # return None
        elif self.readout_op == 'sort':
            x = self.readout(x, batch, self.k)
            x = x.view(len(x), self.k, -1).permute(0, 2, 1)
            x = F.elu(self.sort_conv(x))
            x = x.view(len(x), -1)
            x = self.sort_lin(x)
            return x
        elif self.readout_op == 'mema':
            x1 = global_mean_pool(x, batch)
            x2 = global_max_pool(x, batch)
            x = torch.cat([x1, x2], dim=-1)
            x = self.lin(x)
            return x
        else:
            try:
                x = self.readout(x, batch)
            except:
                print(self.readout_op)
                print('size:', x.size, batch.size())
            if self.readout_op == 'set2set':
                x = self.s2s_lin(x)
            return x