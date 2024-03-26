import torch
import torch.nn as nn

from model.model_variant import build_model_variant


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
        
        attn_dimension = (self.args.conv_layer_num + self.args.short_conv_layer_num) * self.args.dim
        
        self.long_term_gnn, self.short_term_gnn, self.temporal_seq_attn_layer, self.cross_attn_layer = build_model_variant(self.args.model_variant, self.args, self.relation_num)
        
        self.predict_emb_w = nn.Embedding(item_num, attn_dimension, padding_idx=0).to(self.device)
        self.predict_emb_b = nn.Embedding(item_num, 1, padding_idx=0).to(self.device)
        self.node_embeddings = nn.Embedding(node_num, self.args.dim, padding_idx=0).to(self.device)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.predict_emb_w.weight.data.normal_(0, 1.0 / self.predict_emb_w.embedding_dim)
        self.predict_emb_b.weight.data.zero_()
        self.node_embeddings.weight.data.normal_(0, 1.0 / self.node_embeddings.embedding_dim)
        
    def forward(self, X_user_item, X_graph_base, for_pred=False):
        batch_users, batch_sequences, items_to_predict = X_user_item
        edge_index, edge_type, node_no, short_term_part = X_graph_base
        x = self.node_embeddings(node_no)

        rate = torch.tensor([[1] for i in range(edge_type.size()[0])]).to(self.device)

        concat_states = []
        
        x, concat_states = self.long_term_gnn(x, edge_index, edge_type, rate=rate, concat_states=concat_states)
        x, concat_states = self.short_term_gnn(short_term_part, x, rate=rate, concat_states=concat_states)
        
        concat_states = torch.cat(concat_states, dim=1)
        user_embeddings = concat_states[batch_users]
        item_embeddings = concat_states[batch_sequences]
        
        # Temporal Sequential Attention Layer
        item_embeddings = self.temporal_seq_attn_layer(item_embeddings)
        
        if self.args.model_variant in [1, 2]:
            # Cross Attention Layer
            item_embeddings = self.cross_attn_layer(user_emb = user_embeddings, 
                                                    item_emb = item_embeddings)
            
        pe_w = self.predict_emb_w(items_to_predict)
        pe_b = self.predict_emb_b(items_to_predict)
        
        # print('\npe_w:', pe_w.size())
        # print('pe_b:', pe_b.size())
        # print('user_embeddings:', user_embeddings.size())
        # print('item_embeddings:', item_embeddings.size())
        # print('items_to_predict:', items_to_predict.size(), 'for_pred:', for_pred)
        
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