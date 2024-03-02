import torch
import torch.nn as nn

from model.FFN import PointWiseFeedForward, SimpleFeedForward
from model.improvedGNN import LongTermGNN, ShortTermGNN
from model.attention_layers import TemporalSequentialAttentionLayer, CrossAttentionLayer


class CAGSRec(nn.Module):
    """
    Main model class for CAGSRec: Cross-Attention Graph for Sequential Recommendation
    """
    def __init__(self, config, item_num, node_num, relation_num):
        super(CAGSRec, self).__init__()
        
        self.args = config
        self.device = self.args.device
        self.item_num = item_num
        self.node_num = node_num
        self.relation_num = relation_num
        
        attn_dimension = (self.args.conv_layer_num +
                              self.args.short_conv_layer_num) * self.args.dim
        
        if self.args.FFN == 'Simple':
            self.FFN = nn.Sequential(
                SimpleFeedForward(attn_dimension, self.args.attn_drop).to(self.device),
                torch.nn.LayerNorm(attn_dimension, eps=1e-8).to(self.device)
            ).to(self.device)
            
        elif self.args.FFN == 'PointWise':
            self.FFN = nn.Sequential(
                PointWiseFeedForward(attn_dimension, self.args.attn_drop).to(self.device),
                torch.nn.LayerNorm(attn_dimension, eps=1e-8).to(self.device)
            ).to(self.device)
        
        
        self.temporal_seq_attn_layer = TemporalSequentialAttentionLayer(embed_dim=attn_dimension,
                                                                        num_heads=self.args.TSAL_head_num,
                                                                        dropout=self.args.attn_drop,
                                                                        feedforward=self.FFN).to(self.device)
        
        self.cross_attn_layer = CrossAttentionLayer(embed_dim=attn_dimension,
                                                    num_heads=self.args.CAL_head_num,
                                                    dropout=self.args.attn_drop,
                                                    feedforward=self.FFN).to(self.device)
        
        self.long_term_gnn = LongTermGNN(dim = self.args.dim,
                                         conv_layer_num=self.args.conv_layer_num,
                                         relation_num=relation_num,
                                         num_bases=self.args.num_bases).to(self.device)
        
        self.short_term_gnn = ShortTermGNN(dim=self.args.dim,
                                           conv_layer_num=self.args.short_conv_layer_num,
                                           relation_num=relation_num,
                                           num_bases=self.args.num_bases).to(self.device)
        
        self.predict_emb_w = nn.Embedding(item_num, (self.args.conv_layer_num + self.args.short_conv_layer_num) * self.args.dim, padding_idx=0).to(self.device)
        self.predict_emb_b = nn.Embedding(item_num, 1, padding_idx=0).to(self.device)
        self.node_embeddings = nn.Embedding(node_num, self.args.dim, padding_idx=0).to(self.device)
        # print(node_num, self.args.dim, 'nn.emb')
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.predict_emb_w.weight.data.normal_(0, 1.0 / self.predict_emb_w.embedding_dim)
        self.predict_emb_b.weight.data.zero_()
        self.node_embeddings.weight.data.normal_(0, 1.0 / self.node_embeddings.embedding_dim)
        
    def forward(self, X_user_item, X_graph_base, for_pred=False):
        batch_users, batch_sequences, items_to_predict = X_user_item[0], X_user_item[1], X_user_item[2]
        edge_index, edge_type, node_no, short_term_part = X_graph_base[0], X_graph_base[1], X_graph_base[2], X_graph_base[3]
        x = self.node_embeddings(node_no)

        rate = torch.tensor([[1] for i in range(edge_type.size()[0])]).to(self.device)

        concat_states = []

        # Long term GNN
        x, concat_states = self.long_term_gnn(x, edge_index, edge_type, rate=rate, concat_states=concat_states)
        
        # Short term GNN
        x, concat_states = self.short_term_gnn(short_term_part, x, rate=rate, concat_states=concat_states)
        
        concat_states = torch.cat(concat_states, dim=1)
        user_embeddings = concat_states[batch_users]
        item_embeddings = concat_states[batch_sequences]
        
        # Temporal Sequential Attention Layer
        item_embeddings = self.temporal_seq_attn_layer(item_embeddings)
        
        # Cross Attention Layer
        # item_embeddings = self.cross_attn_layer(user_emb = user_embeddings, 
        #                                         item_emb = item_embeddings)
        
        pe_w = self.predict_emb_w(items_to_predict)
        pe_b = self.predict_emb_b(items_to_predict)
        
        if for_pred:
            pe_w = pe_w.squeeze()
            pe_b = pe_b.squeeze()
            # user-pred_item
            res = user_embeddings.mm(pe_w.t()) + pe_b 
            # item-item 
            rel_score = torch.matmul(item_embeddings, pe_w.t().unsqueeze(0)) 
            rel_score = torch.sum(rel_score, dim=1) 
            res += rel_score  
            return res  

        res = torch.baddbmm(pe_b, pe_w, user_embeddings.unsqueeze(2)).squeeze()
        rel_score = item_embeddings.bmm(pe_w.permute(0, 2, 1))
        rel_score = torch.sum(rel_score, dim=1)
        res += rel_score
        
        return res, user_embeddings, item_embeddings