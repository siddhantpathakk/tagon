import torch
import torch.nn as nn
from models.feed_forward import PointWiseFeedForward


class UserAttentionLayer(nn.Module):
    def __init__(self, arg, dim, device):
        super(UserAttentionLayer, self).__init__()
        
        self.arg = arg
        self.user_attn_layer_num = 1
        self.user_attn_layer = nn.ModuleList()
        self.user_attn_head_num = self.arg.user_attn_head_num
        self.user_attn_drop = self.arg.user_attn_drop
        self.user_attn_dim = dim
        self.device = device

        self.module_list = nn.ModuleList()
        
        for i in range(self.user_attn_layer_num):
            self.module_list.append(nn.MultiheadAttention(self.user_attn_dim, self.user_attn_head_num, dropout=self.user_attn_drop).to(self.device))
            
        self.module_list.append(nn.Sequential(
            PointWiseFeedForward(self.user_attn_dim, self.user_attn_drop),
            nn.LayerNorm(self.user_attn_dim, eps=1e-8),
        ))
        
    def forward(self, user_emb):
        user_attn_output = user_emb
        # perform MHA
        for i in range(self.user_attn_layer_num):
            user_attn_output, _ = self.module_list[i](user_attn_output, user_attn_output, user_attn_output)
        # perform FFN
        user_attn_output = self.module_list[-1](user_attn_output)
        
        return user_attn_output