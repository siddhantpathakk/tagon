import torch
import torch.nn as nn
import numpy as np
import logging
import torch.nn.functional as F


from model.merge import MergeLayer
from model.pool import LSTMPool, MeanPool
from model.attention import AttnModel
from model.encoder import TimeEncode, PosEncode, EmptyEncode

class TGAN(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat,
                 attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
                 num_layers=3, n_head=4, null_idx=0, num_heads=1, drop_out=0.1, seq_len=None):
        super(TGAN, self).__init__()
        
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        
        self.feat_dim = self.n_feat_th.shape[1]
        
        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim
        
        self.use_time = use_time
        self.time_att_weights = torch.nn.Parameter(torch.from_numpy(np.random.rand(node_dim, time_dim)).float())

        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        
        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, 
                                                               self.feat_dim, 
                                                               self.feat_dim,
                                                               attn_mode=attn_mode, 
                                                               n_head=n_head, 
                                                               drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')
        
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
        
    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        
 
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        
        return score

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def contrast_nosigmoid(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score, neg_score


    def time_att_aggregate(self, node_emb, node_time_emb):
        """
        node_emb: [batch_size, node_dim] or [batch_size, L, node_dim]
        node_time_emb: [batch_size, components, time_emb]
        """
        batch_size = node_emb.shape[0]

        #[N, L(optional), node_dim] * [node_dim, time_dim] = [N, L(optional), time_dim]
        node_emb_to_time = torch.tensordot(node_emb, self.time_att_weights, dims=([-1], [0]))


        node_emb_to_time = torch.unsqueeze(node_emb_to_time, dim=-2) #[N, L(optional) 1, time_dim]
        if len(node_emb.shape) == 2:
            node_emb_to_time = torch.unsqueeze(node_emb_to_time, dim=-2) #[N, L(optional), 1, 1, time_dim]

        unnormalized_attentions = torch.sum(node_emb_to_time * node_time_emb, dim=-1) #[N, L(optional), components]

        normalized_attentions = F.softmax(unnormalized_attentions, dim=-1) #[N, L(optional), components]
        normalized_attentions = torch.unsqueeze(normalized_attentions, dim=-1) #[N, L(optional), components, 1]

        weighted_time_emb = torch.sum(normalized_attentions * node_time_emb, dim=-2) #[batch_size, L(optional), time_emb]
        return weighted_time_emb



    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20):
        assert(curr_layers >= 0)
        
        device = self.n_feat_th.device
    
        batch_size = len(src_idx_l)
        
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)
        
        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l, 
                                                cut_time_l,
                                                curr_layers=curr_layers - 1, 
                                                num_neighbors=num_neighbors)
                    

            print(src_node_conv_feat.shape, src_node_t_embed.shape)
            src_node_t_embed = self.time_att_aggregate(src_node_t_embed, src_node_t_embed) # do self attn
            
            src_node_t_embed = self.time_att_aggregate(src_node_conv_feat, src_node_t_embed) # do cross attn
            
  
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
                                                                    src_idx_l, 
                                                                    cut_time_l, 
                                                                    num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
            
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            
            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)  
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1, 
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            
            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]
                        
            local, weight = attn_m(src_node_conv_feat, 
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed, 
                                   src_ngn_edge_feat, 
                                   mask)
            return local