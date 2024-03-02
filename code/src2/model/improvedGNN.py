import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math

class ImprovedGNNunit(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases, root_weight=True, bias=True, **kwargs):
        super(ImprovedGNNunit, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att_r = Parameter(torch.Tensor(num_relations, num_bases))
        self.att = Parameter(torch.Tensor(1, 2 * out_channels))
        self.gate_layer = nn.Linear(2*out_channels, 1)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.negative_slope = 0.2
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.0))

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # self.reset_parameters()

    def uniform(size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)
        
    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        self.uniform(size, self.basis)
        self.uniform(size, self.att_r)
        self.uniform(size, self.root)
        self.uniform(size, self.bias)
        self.uniform(size, self.att)

    def forward(self, x, edge_index, edge_type, rate, edge_norm=None, size=None):
        self.rate = rate
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def message(self, x_j, edge_index_i, x_i, edge_type, size_i, edge_norm):
        w = torch.matmul(self.att_r, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        w = torch.index_select(w, 0, edge_type)

        if x_j is not None:
            x_j = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        if self.root is not None and x_i is not None:
            x_i = torch.matmul(x_i, self.root)

        alpha = self.compute_attention(x_i, x_j, edge_index_i, size_i)

        out = x_j * alpha.view(-1, 1)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def compute_attention(self, x_i, x_j, edge_index_i, size_i):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index=edge_index_i, num_nodes=size_i)
        return self.dropout(alpha)

    def update(self, aggr_out, x):
        if self.root is not None and x is not None:
            aggr_out += torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out += self.bias

        return self.layer_norm(aggr_out)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)
        
class LongTermGNN(nn.Module):
    
    def __init__(self, dim, conv_layer_num, relation_num, num_bases, conv_latent_dim=None):
        super(LongTermGNN, self).__init__()
        
        self.conv_latent_dim = [dim for _ in range(conv_layer_num)]
        self.conv_modulelist = torch.nn.ModuleList()
        
        self.conv_modulelist.append(ImprovedGNNunit(dim,self.conv_latent_dim[0],relation_num, num_bases))
        for i in range(len(self.conv_latent_dim)-1):
            self.conv_modulelist.append(ImprovedGNNunit(self.conv_latent_dim[i], self.conv_latent_dim[i+1], relation_num, num_bases, ))
                
    def forward(self, x, edge_index, edge_type, rate=0.0, concat_states=[]):
        for conv in self.conv_modulelist:
            x = torch.tanh(conv(x, edge_index, edge_type, rate=rate))
            concat_states.append(x)
        
        return x, concat_states
    
class ShortTermGNN(nn.Module):
    
    def __init__(self, dim, conv_layer_num, relation_num, num_bases, conv_latent_dim=None):
        super(ShortTermGNN, self).__init__()

        self.conv_latent_dim = [dim for _ in range(conv_layer_num)]

        self.short_conv_modulelist = torch.nn.ModuleList()
        self.short_conv_modulelist.append(ImprovedGNNunit(dim, self.conv_latent_dim[0], relation_num, num_bases))

        for i in range(len(self.conv_latent_dim)-1):
            self.short_conv_modulelist.append(ImprovedGNNunit(self.conv_latent_dim[i], self.conv_latent_dim[i+1], relation_num, num_bases))
        
    def forward(self, short_term_part, x, rate=0.0, concat_states=[]):
        for conv in self.short_conv_modulelist:
            for i in range(len(short_term_part)):
                short_edge_index,short_edge_type = short_term_part[i][0],short_term_part[i][1]
                x = torch.tanh(conv(x, short_edge_index, short_edge_type, rate=rate))
            concat_states.append(x)
        
        return x, concat_states