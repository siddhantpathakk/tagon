import logging
import torch
from model.components.ffn import MergeLayer
from model.components.attention import MultiHeadAttention, MapBasedMultiHeadAttention

class BaseAttentionModel(torch.nn.Module):
    def __init__(self, feat_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
      super(BaseAttentionModel, self).__init__()
      
      self.logger = logging.getLogger(__name__)
      self.attn_mode = attn_mode
      self.n_head = n_head
      self.drop_out = drop_out
      
      self.model_dim = feat_dim + time_dim
      self.feat_dim = feat_dim
      self.time_dim = time_dim

      self.ffn = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
      
      assert(self.model_dim % n_head == 0)
      
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
       raise NotImplementedError('forward method must be implemented in the derived class')


class CrossAttentionModel(BaseAttentionModel):
    def __init__(self, feat_dim, time_dim, attn_mode, n_head, drop_out):
        """
        args:
          feat_dim: dim for the node features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(CrossAttentionModel, self).__init__(feat_dim, time_dim, attn_mode, n_head, drop_out)
        

    def forward(self, src, src_t, seq, seq_t, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        q = torch.cat([src_ext, src_t], dim=2) # [B, 1, D + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_t], dim=2) # [B, 1, D + Dt] -> [B, 1, D]
        
        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()
        attn = attn.squeeze()

        output = self.ffn(output, src)
        return output, attn


class SelfAttentionModel(BaseAttentionModel):
    def __init__(self, time_dim, attn_mode="prod", n_head=2, drop_out=0.1):
        super(SelfAttentionModel, self).__init__(time_dim, time_dim, attn_mode, n_head, drop_out)
      
    def forward(self, seq, seq_t, mask):
          # TODO: implement the forward pass for self attention
          src = torch.squeeze(seq, dim=1)
          src_t = torch.squeeze(seq_t, dim=1)
          # src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
          q = torch.cat([src, src_t], dim=2) # [B, 1, D + Dt] -> [B, 1, D]
          k = torch.cat([seq, seq_t], dim=2) # [B, 1, D + Dt] -> [B, 1, D]
          
          mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
          mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

          # # target-attention
          output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
          output = output.squeeze()
          attn = attn.squeeze()

          output = self.ffn(output, src)
          return output, attn