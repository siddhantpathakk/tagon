import logging
import torch

from src.model.layers.attention.utils import MapBasedMultiHeadAttention, MultiHeadAttention
from src.model.layers.ffn import MergeLayer, MergeSelfAttnLayer
torch.cuda.empty_cache()

class BaseAttentionLayer(torch.nn.Module):
    def __init__(self, args, feat_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        self.feat_dim = feat_dim
        self.time_dim = time_dim

        self.model_dim = (feat_dim + time_dim)
 
        assert (self.model_dim % n_head == 0)
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


    def forward(self):
        raise NotImplementedError('forward method must be implemented')
        
    
class SequentialSelfAttention(BaseAttentionLayer):
    def __init__(self, args, feat_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        
        super(SequentialSelfAttention, self).__init__(args, feat_dim, time_dim, attn_mode, n_head, drop_out)
        self.ffn = MergeSelfAttnLayer(feat_dim, time_dim, batch_size=args.bs, n=args.n_degree)

    def forward(self, seq, seq_t, mask):

        # reverse the sequence
        # seq_rev = torch.flip(seq, dims=[-1])
        # seq_t_rev = torch.flip(seq_t, dims=[-1])
        # k = torch.cat([seq, seq_rev], dim=2)
        # k_rev = torch.cat([seq_rev, seq_t_rev], dim=2)
        
        k = torch.cat([seq, seq_t], dim=2)

        output, attn = self.multi_head_target(q=k, k=k, v=k, mask=mask)
        output = self.ffn(output, seq)

        return output, attn


class TemporalCrossAttention(BaseAttentionLayer):
    def __init__(self, args, feat_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):

        super(TemporalCrossAttention, self).__init__(args, feat_dim, time_dim, attn_mode, n_head, drop_out)
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, k, q, v, mask):
        output, attn = self.multi_head_target(q=q, k=k, v=v, mask=mask)
        output = output.squeeze()
        attn = attn.squeeze()
        output = self.merger(output, src)
        return output, attn


class Transformer(torch.nn.Module):
    def __init__(self, args, feat_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        super(Transformer, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.model_dim = (feat_dim + time_dim)

        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        self.self_attn_model = SequentialSelfAttention(args, feat_dim, time_dim, attn_mode, n_head, drop_out)
        self.cross_attn_model = TemporalCrossAttention(args, feat_dim, time_dim, attn_mode, n_head, drop_out)
        
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
        src_ext = torch.unsqueeze(src, dim=1) 
        q = torch.cat([src_ext, src_t], dim=2) 
        k, _ = self.self_attn_model(seq, seq_t, mask)

        mask = torch.unsqueeze(mask, dim=2)
        mask = mask.permute([0, 2, 1]) 

        return self.cross_attn_model(src, k, q, q, mask)