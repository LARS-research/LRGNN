from operations import *
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, BatchNorm
# def act_map(act):
#     if act == "linear":
#         return lambda x: x
#     elif act == "elu":
#         return torch.nn.functional.elu
#     elif act == "sigmoid":
#         return torch.sigmoid
#     elif act == "tanh":
#         return torch.tanh
#     elif act == "relu":
#         return torch.nn.functional.relu
#     elif act == "relu6":
#         return torch.nn.functional.relu6
#     elif act == "softplus":
#         return torch.nn.functional.softplus
#     elif act == "leaky_relu":
#         return torch.nn.functional.leaky_relu
#     else:
#         raise Exception("wrong activate function")

class NaOp(nn.Module):
    def __init__(self, primitive, in_dim, out_dim, with_linear=False):
        super(NaOp, self).__init__()

        self._op = NA_OPS[primitive](in_dim, out_dim)
        self.op_linear = nn.Linear(in_dim, out_dim)
        self.with_linear = with_linear

    def reset_parameters(self):
        self._op.reset_parameters()
        self.op_linear.reset_parameters()

    def forward(self, x, edge_index):
        if self.with_linear:
            return self._op(x, edge_index) + self.op_linear(x)
        else:
            return self._op(x, edge_index)


class ScOp(nn.Module):
    def __init__(self, primitive):
        super(ScOp, self).__init__()
        self._op = SC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)

class LaOp(nn.Module):
    def __init__(self, primitive, hidden_size,num_layers=None):
        super(LaOp, self).__init__()
        self._op = FF_OPS[primitive](hidden_size, num_layers)
    def reset_parameters(self):
        self._op.reset_parameters()
    def forward(self, x):
        return F.relu(self._op(x))

class ReadoutOp(nn.Module):
    def __init__(self, primitive, hidden):
        super(ReadoutOp, self).__init__()
        self._op = READOUT_OPS[primitive](hidden)
    def reset_parameters(self):
        self._op.reset_parameters()
    def reset_params(self):
        self._op.reset_params()

    def forward(self, x, batch):
        return self._op(x, batch)

class NetworkGNN(nn.Module):

    def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size, dropout=0.5, act='relu', args=None):
        super(NetworkGNN, self).__init__()
        self.genotype = genotype
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self._criterion = criterion

        self.num_blocks = args.num_blocks
        self.num_cells = args.num_cells
        self.cell_mode = args.cell_mode

        ops = genotype.split('||')
        self.args = args

        # pre-process
        self.lin1 = nn.Linear(in_dim, hidden_size)

        # aggregation
        self.gnn_layers = nn.ModuleList(
            [NaOp(ops[i], hidden_size, hidden_size) for i in range(self.num_blocks)])

        # selection
        num_node_per_cell = int(self.num_blocks / self. num_cells)
        self.num_node_per_cell = num_node_per_cell

        if self.cell_mode == 'full':
            num_searched_skip = (self.args.num_blocks + 2) * (self.args.num_blocks + 1) / 2
        # elif self.cell_mode == 'repeat':
        #     num_searched_skip = (num_node_per_cell + 2) * (num_node_per_cell + 1) / 2
        else: # diverse or repeat
            num_searched_skip = self.num_cells * (num_node_per_cell + 2) * (num_node_per_cell + 1) / 2

        self.num_edges = int(num_searched_skip)
        self.skip_op = nn.ModuleList()
        for i in range(self.num_edges):
            self.skip_op.append(ScOp(ops[self.num_blocks + i]))

        # fuse function
        self.fuse_funcs = nn.ModuleList()
        start = self.num_edges + self.num_blocks
        for i in range(self.num_blocks + self.num_cells):
            if self.cell_mode == 'full':
                input_blocks = i + 1
            else:
                input_blocks = i % (num_node_per_cell + 1) + 1
            self.fuse_funcs.append(LaOp(ops[start + i], hidden_size, num_layers=input_blocks))

        self.cell_output_lins = nn.ModuleList()
        for i in range(self.num_cells):
            self.cell_output_lins.append(Linear(hidden_size, hidden_size))

        self.readout_layers = ReadoutOp(ops[-1], hidden_size)
        self.readout_lin = Linear(hidden_size, hidden_size)
        self.classifier = Linear(hidden_size, out_dim)

        #extra ops
        self.lns = nn.ModuleList()
        if self.args.LN:
            for i in range(self.num_blocks):
                self.lns.append(LayerNorm(hidden_size))

        self.bns = nn.ModuleList()
        if self.args.BN:
            for i in range(self.num_blocks):
                self.bns.append(BatchNorm(hidden_size))

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
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        
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

        # if self.cell_mode == 'repeat':
        #     return cur_node
        # else: #diverse or full
        #     return cell * (self.num_node_per_cell + 1) + cur_node
        return cell * (self.num_node_per_cell + 1) + cur_node


    def forward(self, data):
        cell_output = []
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        features = []

        # input node 0
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        features += [x]
        cell_output += [x]
        num_node_per_cell = int(self.num_blocks / self.num_cells)
        for cell in range(self.num_cells):
            for node in range(num_node_per_cell + 1):
                # select inputs
                layer_input = []
                for i in range(node + 1):
                    edge_id = self._get_edge_id(cell, node, i)
                    layer_input += [self.skip_op[edge_id](features[i])]

                # fuse features
                ff_id = self._get_ff_id(cell, node)
                tmp_input = self.fuse_funcs[ff_id](layer_input)

                # aggregation
                agg_id = cell * self.num_node_per_cell + node
                if node == self.num_node_per_cell:
                    x = self.cell_output_lins[cell](tmp_input)
                else:
                    x = self.gnn_layers[agg_id](tmp_input, edge_index)
                x = F.relu(x)

                if node != self.num_node_per_cell:
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

        output = self.readout_layers(x, batch)
        output = F.relu(self.readout_lin(output))
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.classifier(output)
        return output






