import torch
# torch.cuda.empty_cache()
import torch.nn.functional as F
import logging
import numpy as np

from src.components.models.encoders import DisentangleTimeEncode, EmptyEncode, PosEncode, TimeEncode
from src.components.models.attention import AttnModel
from src.components.models.layers import MergeLayer
from src.components.models.pooling import LSTMPool, MeanPool



class TAGON(torch.nn.Module):
    """
    Main model class for TAGON: Temporal Attention Graph Optimized Network.

    Parameters:
        ngh_finder: object, the neighbor finder object.
        n_nodes: int, the number of nodes.
        args: object, the arguments object.
        attn_mode: str, the attention mode.
        use_time: str, the time mode.
        agg_method: str, the aggregation method.
        node_dim: int, the node dimension.
        time_dim: int, the time dimension.
        num_layers: int, the number of layers.
        n_head: int, the number of heads.
        null_idx: int, the null index.
        num_heads: int, the number of heads.
        drop_out: float, the dropout rate.
        seq_len: int, the sequence length.
    """
    def __init__(self, ngh_finder, 
                 n_nodes, 
                 args,
                 attn_mode='prod', 
                 use_time='time', 
                 agg_method='attn', 
                 node_dim=32, 
                 time_dim=32,
                 num_layers=3, 
                 n_head=4, 
                 null_idx=0, 
                 num_heads=1, 
                 drop_out=0.1, 
                 seq_len=None):
        
        super(TAGON, self).__init__()
        
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)
        self.feat_dim = node_dim
        self.use_time = use_time
        self.n_feat_dim = node_dim
        self.model_dim = self.n_feat_dim + time_dim
        self.use_time = use_time
        self.node_hist_embed = torch.nn.Embedding(n_nodes, node_dim)
        self.time_att_weights = torch.nn.Parameter(torch.from_numpy(np.random.rand(node_dim, time_dim)).float())
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        
        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(args, self.n_feat_dim, 
                                                                    time_dim,
                                                                    attn_mode=attn_mode, 
                                                                    n_head=n_head, 
                                                                    drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.n_feat_dim,
                                                                 time_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.n_feat_dim) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=time_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=time_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=time_dim)
        elif use_time == 'disentangle':
            self.logger.info('Using disentangle time encoding')
            self.time_encoder = DisentangleTimeEncode(args.disencomponents, expand_dim=time_dim)
        else:
            raise ValueError('invalid time option!')
        

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1) 
        
        torch.nn.init.uniform_(self.node_hist_embed.weight, a=-1.0, b=1.0)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        
        return score

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        """
        Probability calculation for contrastive learning, with sigmoid.

        Parameters:
            src_idx_l: list, the source node indices.
            target_idx_l: list, the target node indices.
            background_idx_l: list, the background node indices.
            cut_time_l: list, the cut times.
            num_neighbors: int, the number of neighbors.

        Returns:
            pos_score: torch.Tensor, the positive probability.
            neg_score: torch.Tensor, the negative probability.
        """
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def contrast_nosigmoid(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        """
        Score calculation for contrastive learning without sigmoid.

        Parameters:
            src_idx_l: list, the source node indices.
            target_idx_l: list, the target node indices.
            background_idx_l: list, the background node indices.
            cut_time_l: list, the cut times.
            num_neighbors: int, the number of neighbors.

        Returns:
            pos_score: torch.Tensor, the positive score.
            neg_score: torch.Tensor, the negative score.
        """
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score, neg_score


    def time_att_aggregate(self, node_emb, node_time_emb):
        """
        Aggregate the time embeddings based on the node embeddings.

        Parameters:
            node_emb: torch.Tensor, the node embeddings.
            node_time_emb: torch.Tensor, the time embeddings.

        Returns:
            weighted_time_emb: torch.Tensor, the weighted time embeddings.
        """

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
        """
        Temporal convolutional layer.

        Parameters:
            src_idx_l: list, the source node indices.
            cut_time_l: list, the cut times.
            curr_layers: int, the current layer.
            num_neighbors: int, the number of neighbors.

        Returns:
            local: torch.Tensor, the local features.
        """
        assert(curr_layers >= 0)
        
        device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
        
        #if macbook, use mps
        # try:
        #     device = torch.device('mps')
        # except:
        #     pass
    
        batch_size = len(src_idx_l)
        
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_hist_embed(src_node_batch_th)
        
        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l, 
                                           cut_time_l,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors)
            
            if self.use_time == 'disentangle':
                src_node_t_embed = self.time_att_aggregate(src_node_conv_feat, src_node_t_embed)
            
            src_ngh_node_batch, _, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
                                                                    src_idx_l, 
                                                                    cut_time_l, 
                                                                    num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            
            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1, 
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            
            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            if self.use_time == 'disentangle':
                src_ngh_t_embed = self.time_att_aggregate(src_ngh_feat, src_ngh_t_embed)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]
                        
            local, _ = attn_m(src_node_conv_feat, 
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed, 
                                   mask)
            return local
