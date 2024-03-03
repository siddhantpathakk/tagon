import torch
import torch.nn as nn

from model.FFN import PointWiseFeedForward, SimpleFeedForward
from model.improvedGNN import LongTermGNN, ShortTermGNN
from model.attention_layers import TemporalSequentialAttentionLayer_v2, CrossAttentionLayer_v2


class CAGSRec(nn.Module):
    """
    Main model class for CAGSRec: Cross-Attention Graph for Sequential Recommendation
    """
    def __init__(self, config, item_num, node_num, relation_num, gcn):
        super(CAGSRec, self).__init__()
        
        self.args = config
        self.device = self.args.device
        self.item_num = item_num
        self.node_num = node_num
        self.relation_num = relation_num
        
        attn_dimension = (self.args.conv_layer_num + self.args.short_conv_layer_num) * self.args.dim
        
        if self.args.FFN == 'Simple':
            self.FFN = nn.Sequential(
                nn.Linear(attn_dimension, attn_dimension).to(self.device), 
                # SimpleFeedForward(attn_dimension, self.args.attn_drop, device=self.device).to(self.device), 
                torch.nn.LayerNorm(attn_dimension, eps=1e-8).to(self.device),
            ).to(self.device)

        elif self.args.FFN == 'PointWise':
            self.FFN = nn.Sequential(
                PointWiseFeedForward(attn_dimension, self.args.attn_drop, device=self.device).to(self.device),
                torch.nn.LayerNorm(attn_dimension, eps=1e-8).to(self.device)
            ).to(self.device)
        
        self.long_term_gnn = LongTermGNN(dim=self.args.dim, 
                                         conv_layer_num=self.args.conv_layer_num, 
                                         relation_num=relation_num,
                                         num_bases=self.args.num_bases, 
                                         device=self.device).to(self.device)
        

        self.short_term_gnn = ShortTermGNN(dim=self.args.dim, 
                                         conv_layer_num=self.args.conv_layer_num, 
                                         relation_num=relation_num,
                                         num_bases=self.args.num_bases, 
                                         device=self.device).to(self.device)

        
        self.temporal_seq_attn_layer = TemporalSequentialAttentionLayer_v2(attn_dim=attn_dimension,
                                                                            num_heads=self.args.TSAL_head_num,
                                                                            feedforward=self.FFN,
                                                                            dropout=self.args.attn_drop,
                                                                            device=self.device).to(self.device)
        
        self.cross_attn_layer = CrossAttentionLayer_v2(attn_dim=attn_dimension,
                                                        num_heads=self.args.CAL_head_num,
                                                        feedforward=self.FFN,
                                                        dropout=self.args.attn_drop,
                                                        device=self.device).to(self.device)
        
        # # gcn (long term)
        # conv_latent_dim = [self.args.dim for _ in range(self.args.conv_layer_num)]
        # self.conv_modulelist = torch.nn.ModuleList()
        # self.conv_modulelist.append(gcn(self.args.dim,conv_latent_dim[0],relation_num,self.args.num_bases,device=self.device))
        # for i in range(len(conv_latent_dim)-1):
        #     self.conv_modulelist.append(gcn(conv_latent_dim[i],conv_latent_dim[i+1],relation_num,self.args.num_bases,device=self.device))
        
        # # gcn (short term)
        # short_conv_latent_dim = [self.args.dim for _ in range(self.args.short_conv_layer_num)]
        # self.short_conv_modulelist = torch.nn.ModuleList()
        # self.short_conv_modulelist.append(gcn(self.args.dim,short_conv_latent_dim[0],relation_num,self.args.num_bases,device=self.device))
        # for i in range(len(short_conv_latent_dim)-1):
        #     self.short_conv_modulelist.append(gcn(short_conv_latent_dim[i],short_conv_latent_dim[i+1],relation_num,self.args.num_bases,device=self.device))
        
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
        
        # for conv in self.conv_modulelist:
        #     x = torch.tanh(conv(x, edge_index, edge_type, gate_emd2=None, rate=rate))
        #     concat_states.append(x)
        
        # for conv in self.short_conv_modulelist:
        #     for i in range(len(short_term_part)):
        #         short_edge_index,short_edge_type = short_term_part[i][0],short_term_part[i][1]
        #         x = torch.tanh(conv(x, short_edge_index, short_edge_type, gate_emd2=None, rate=rate))
        #     concat_states.append(x)
        
        x, concat_states = self.long_term_gnn(x, edge_index, edge_type, rate=rate, concat_states=concat_states)
        x, concat_states = self.short_term_gnn(short_term_part, x, rate=rate, concat_states=concat_states)
        
        concat_states = torch.cat(concat_states, dim=1)
        user_embeddings = concat_states[batch_users]
        item_embeddings = concat_states[batch_sequences]
        
        # Temporal Sequential Attention Layer
        item_embeddings = self.temporal_seq_attn_layer(item_embeddings)
        
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