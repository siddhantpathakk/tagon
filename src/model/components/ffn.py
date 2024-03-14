import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)
    

class MergeSelfAttnLayer(nn.Module):
    def __init__(self, feat_dim, time_dim, batch_size, n):
        super(MergeSelfAttnLayer, self).__init__()
        self.fc1 = nn.Linear(feat_dim+time_dim+feat_dim, batch_size * n)
        self.fc2 = nn.Linear(batch_size * n, feat_dim+time_dim)
        self.act = nn.ReLU()            
        
    def forward(self, seq, seq_t):
        x = torch.cat([seq, seq_t], dim=-1)
        bs, n, _ = x.size()
        x = x.view(bs * n, -1)
        x = self.fc2(self.act(self.fc1(x)))
        x = x.view(bs, n, -1)
        return x
    