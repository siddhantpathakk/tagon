import torch
from torch.nn import Parameter as Param
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math

class ImprovedGNNunit(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_bases,device, root_weight=True, bias=True, **kwargs):
        super(ImprovedGNNunit, self).__init__(aggr='add', **kwargs)
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels)).to(self.device)
        self.att_r = Param(torch.Tensor(num_relations, num_bases)).to(self.device)
        self.heads = 1
        self.att = Param(torch.Tensor(1, self.heads, 2 * out_channels)).to(self.device)
        self.gate_layer = nn.Linear(2*out_channels, 1)
        self.relu = nn.ReLU()
        self.negative_slope = 0.2

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels)).to(self.device)
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels)).to(self.device)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.dropout = 0

    def uniform(self, size, tensor):
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

    def forward(self, x, edge_index, edge_type, gate_emd2, rate, edge_norm=None, size=None):
        if gate_emd2 is not None:
            self.gate_emd2 = gate_emd2
        else:
            self.gate_emd2 = None
        if rate is not None:
            self.rate = rate
        else:
            print('[ERROR]: rate is empty.')
            
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_index_i, x_i,edge_type,size_i, edge_norm):
        '''
        out shape(6,32)
        w1 (it is in if else;) shape(8,16,32)
        w2 (it is in if else;) shape(6,16,32)
        x_j shape(6,16)
        edge_index_j = 0,0,0,1,2,3
        '''
        w = torch.matmul(self.att_r, self.basis.view(self.num_bases, -1)).to(self.device)

        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            x_j = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        ## thelta_root * x_i
        if self.root is not None:
            if x_i is None:
                x_i = self.root
            else:
                x_i = torch.matmul(x_i, self.root)

        ## attention_ij
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index=edge_index_i, num_nodes=size_i)
        self.attn_weight =  alpha

        out = x_j * alpha.view(-1, self.heads, 1)

        out = out.view(-1,self.out_channels)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        '''
        x shape:(4,16)
        aggr_out shape:(4,32)
        '''
        if self.root is not None:
            if x is None:
                aggr_out = aggr_out + self.root
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
            
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)
        

class BaseGNN(nn.Module):
    
    def __init__(self, dim, conv_layer_num, relation_num, num_bases, device, conv_latent_dim=None):
        super(BaseGNN, self).__init__()
        
        self.device = device 
        
        self.conv_latent_dim = [dim for _ in range(conv_layer_num)]
        self.conv_modulelist = torch.nn.ModuleList()
        
        self.conv_modulelist.append(ImprovedGNNunit(dim,self.conv_latent_dim[0],relation_num, num_bases, device=self.device))
        for i in range(len(self.conv_latent_dim)-1):
            self.conv_modulelist.append(ImprovedGNNunit(self.conv_latent_dim[i], self.conv_latent_dim[i+1], relation_num, num_bases, device=self.device))
            
    def forward(self):
        raise NotImplementedError("forward method is not implemented")


class LongTermGNN(BaseGNN):
    
    def __init__(self, dim, conv_layer_num, relation_num, num_bases, device, conv_latent_dim=None):
        super(LongTermGNN, self).__init__(dim, conv_layer_num, relation_num, num_bases, device, conv_latent_dim)
    
    def forward(self, x, edge_index, edge_type, rate=0.0, concat_states=[], gate_emd2=None):
        for conv in self.conv_modulelist:
            x = torch.tanh(conv(x, edge_index, edge_type, rate=rate, gate_emd2=gate_emd2))
            concat_states.append(x)
        
        return x, concat_states
    
    
class ShortTermGNN(BaseGNN):
    
    def __init__(self, dim, conv_layer_num, relation_num, num_bases, device, short_conv_latent_dim=None):
        super(ShortTermGNN, self).__init__(dim, conv_layer_num, relation_num, num_bases, device, short_conv_latent_dim)
    
    def forward(self, short_term_part, x, rate=0.0, concat_states=[], gate_emd2=None):
        for conv in self.conv_modulelist:
            for i in range(len(short_term_part)):
                short_edge_index, short_edge_type = short_term_part[i][0],short_term_part[i][1]
                x = torch.tanh(conv(x, short_edge_index, short_edge_type, rate=rate, gate_emd2=gate_emd2))
            concat_states.append(x)
        
        return x, concat_states