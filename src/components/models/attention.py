import torch
# torch.cuda.empty_cache()
import torch.nn as nn
import logging
import numpy as np

from src.components.models.layers import MergeLayer, MergeSelfAttnLayer

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention module 
    This module is never used directly, but is used in MultiHeadAttention module.

    Parameters:
        temperature: float, temperature
        attn_dropout: float, dropout rate
    
    Inputs:
        q: float Tensor of shape [n * b, l_q, d]
        k: float Tensor of shape [n * b, l_k, d]
        v: float Tensor of shape [n * b, l_v, d]
        mask: boolean Tensor of shape [n * b, l_q, l_k]

    Returns:
        output: float Tensor of shape [n * b, l_q, d]
    '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
        output = torch.bmm(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module using scaled dot-product attention 
    
    Parameters:
        n_head: int, number of heads
        d_model: int, model dimension
        d_k: int, key dimension
        d_v: int, value dimension
        dropout: float, dropout rate

    Inputs:
        q: float Tensor of shape [n * b, l_q, d]
        k: float Tensor of shape [n * b, l_k, d]
        v: float Tensor of shape [n * b, l_v, d]
        mask: boolean Tensor of shape [n * b, l_q, l_k]

    Returns:
        output: float Tensor of shape [n * b, l_q, d]
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        return output, attn
    

class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module using scaled dot-product attention based on a mapping function
    
    Parameters:
        n_head: int, number of heads
        d_model: int, model dimension
        d_k: int, key dimension
        d_v: int, value dimension
        dropout: float, dropout rate

    Inputs:
        q: float Tensor of shape [n * b, l_q, d]
        k: float Tensor of shape [n * b, l_k, d]
        v: float Tensor of shape [n * b, l_v, d]
        mask: boolean Tensor of shape [n * b, l_q, l_k]

    Returns:
        output: float Tensor of shape [n * b, l_q, d]
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)
        
        nn.init.xavier_normal_(self.fc.weight)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]
        
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]
        
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk
        
        ## Map based Attention
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]
        
        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn
    


class SelfAttn(torch.nn.Module):
    """
    Self attention layer for sequential data

    Parameters:
        args: arguments for the model
        feat_dim: int, feature dimension
        time_dim: int, time dimension
        attn_mode: str, attention mode, 'prod' or 'map'
        n_head: int, number of heads
        drop_out: float, dropout rate

    Inputs:
        seq: float Tensor of shape [B, N, D]
        seq_t: float Tensor of shape [B, N, Dt]
        mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

    Returns:
        output: float Tensor of shape [B, N, D]
        attn: float Tensor of shape [B, N, N]
    """
    def __init__(self, args, feat_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        super(SelfAttn, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.model_dim = (feat_dim + time_dim)

        self.ffn = MergeSelfAttnLayer(feat_dim, time_dim, batch_size=args.bs, n=args.n_degree)
        
        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
        
    def forward(self, seq, seq_t, mask):
        k = torch.cat([seq, seq_t], dim=2)
        mask = torch.unsqueeze(mask, dim=2)
        mask = mask.permute([0, 2, 1])
        output, attn = self.multi_head_target(q=k, k=k, v=k, mask=mask)
        output = self.ffn(output, seq)
        return output, attn 

class AttnModel(torch.nn.Module):
    """Attention based temporal layers

    Parameters:
        args: arguments for the model
        feat_dim: int, feature dimension
        time_dim: int, time dimension
        attn_mode: str, attention mode, 'prod' or 'map'
        n_head: int, number of heads
        drop_out: float, dropout rate

    Inputs:
        src: float Tensor of shape [B, D]
        src_t: float Tensor of shape [B, Dt]
        seq: float Tensor of shape [B, N, D]
        seq_t: float Tensor of shape [B, N, Dt]
        mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

    Returns:
        output: float Tensor of shape [B, D]
        attn: float Tensor of shape [B, N]
    """
    def __init__(self, args, feat_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.model_dim = (feat_dim + time_dim)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        
        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        
        self.self_attn_model = SelfAttn(args, feat_dim, time_dim, attn_mode, n_head, drop_out)
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
        
    def forward(self, src, src_t, seq, seq_t, mask):
        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        q = torch.cat([src_ext, src_t], dim=2) # [B, 1, D + Dt] -> [B, 1, D]
        k, _ = self.self_attn_model(seq, seq_t, mask)        

        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()
        attn = attn.squeeze()

        output = self.merger(output, src)
        return output, attn

