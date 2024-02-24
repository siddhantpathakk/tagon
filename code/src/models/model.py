import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.feed_forward import PointWiseFeedForward

class GNN_SR_Net(nn.Module):
    def __init__(self, config, item_num, node_num, relation_num, gcn, device):
        super(GNN_SR_Net, self).__init__()
                
        # parameter setting
        self.arg = config
        self.device = device
        
        dim = self.arg.dim
        conv_layer_num = self.arg.conv_layer_num
        short_term_conv_layer_num = self.arg.short_term_conv_layer_num
        
        self.adj_dropout = self.arg.adj_dropout
        self.num_bases = self.arg.num_bases
        self.lambda_val = self.arg.lambda_val

        # prediction variable embeddings
        self.predict_emb_w = nn.Embedding(item_num, (conv_layer_num + short_term_conv_layer_num) * dim, padding_idx=0).to(device)
        self.predict_emb_b = nn.Embedding(item_num, 1, padding_idx=0).to(device)
        self.node_embeddings = nn.Embedding(node_num, dim, padding_idx=0).to(device)

        # gcn (long term)
        conv_latent_dim = [dim for _ in range(conv_layer_num)]
        self.conv_modulelist = torch.nn.ModuleList()
        self.conv_modulelist.append(gcn(dim,conv_latent_dim[0],relation_num,self.num_bases,device=self.device))
        for i in range(len(conv_latent_dim)-1):
            self.conv_modulelist.append(gcn(conv_latent_dim[i],conv_latent_dim[i+1],relation_num,self.num_bases,device=self.device))
        
        # gcn (short term)
        short_conv_latent_dim = [dim for _ in range(short_term_conv_layer_num)]
        self.short_conv_modulelist = torch.nn.ModuleList()
        self.short_conv_modulelist.append(gcn(dim,short_conv_latent_dim[0],relation_num,self.num_bases,device=self.device))
        for i in range(len(short_conv_latent_dim)-1):
            self.short_conv_modulelist.append(gcn(short_conv_latent_dim[i],short_conv_latent_dim[i+1],relation_num,self.num_bases,device=self.device))

        # TSAL setting
        self.TSAL_dim = (conv_layer_num+short_term_conv_layer_num) * dim
        self.head_num = self.arg.TSAL_head_num
        self.attn_drop = self.arg.TSAL_attn_drop
        self.drop_layer = nn.Dropout(p=self.attn_drop)
        
        self.TSAL_W_Q = Variable(torch.zeros(self.TSAL_dim,self.TSAL_dim).type(torch.FloatTensor), requires_grad=True).to(device)
        self.TSAL_W_K = Variable(torch.zeros(self.TSAL_dim,self.TSAL_dim).type(torch.FloatTensor), requires_grad=True).to(device)
        self.TSAL_W_V = Variable(torch.zeros(self.TSAL_dim,self.TSAL_dim).type(torch.FloatTensor), requires_grad=True).to(device)

        self.feedforward = nn.Sequential(
            PointWiseFeedForward(self.TSAL_dim, self.attn_drop, self.device), # (default) nn.Sequential(nn.Linear(self.TSAL_dim, self.TSAL_dim))
            torch.nn.LayerNorm(self.TSAL_dim, eps=1e-8).to(self.device),
        )   
        
        # Cross Attention Layer
        self.CAL_dim = (conv_layer_num+short_term_conv_layer_num) * dim
        self.CAL_head_num = self.arg.cross_attn_head_num
        self.CAL_drop = self.arg.cross_attn_drop
        self.CAL_drop_layer = nn.Dropout(p=self.attn_drop)
        
        self.CAL_W_Q = Variable(torch.zeros(self.CAL_dim,self.CAL_dim).type(torch.FloatTensor), requires_grad=True).to(device)
        self.CAL_W_K = Variable(torch.zeros(self.CAL_dim,self.CAL_dim).type(torch.FloatTensor), requires_grad=True).to(device)
        self.CAL_W_V = Variable(torch.zeros(self.CAL_dim,self.CAL_dim).type(torch.FloatTensor), requires_grad=True).to(device)

        self.feedforward_cross = nn.Sequential(
            PointWiseFeedForward(self.CAL_dim, self.CAL_drop, self.device), # (default) nn.Sequential(nn.Linear(self.CAL_dim, self.CAL_dim))
            torch.nn.LayerNorm(self.CAL_dim, eps=1e-8).to(self.device),
        )
     
        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.predict_emb_w.weight.data.normal_(0, 1.0 / self.predict_emb_w.embedding_dim)
        self.predict_emb_b.weight.data.zero_()
        self.node_embeddings.weight.data.normal_(0, 1.0 / self.node_embeddings.embedding_dim)
        
        self.TSAL_W_Q = nn.init.xavier_uniform_(self.TSAL_W_Q)
        self.TSAL_W_K = nn.init.xavier_uniform_(self.TSAL_W_K)
        self.TSAL_W_V = nn.init.xavier_uniform_(self.TSAL_W_V)
        
        self.CAL_W_Q = nn.init.xavier_uniform_(self.CAL_W_Q)
        self.CAL_W_K = nn.init.xavier_uniform_(self.CAL_W_K)
        self.CAL_W_V = nn.init.xavier_uniform_(self.CAL_W_V)
        
    def Temporal_Attention_Layer(self,input_tensor):
        time_step = input_tensor.size()[1]
        
        # positional embedding using sin and cos function
        pos_emb = torch.arange(0, time_step).unsqueeze(0).repeat(input_tensor.size()[0],1).to(self.device) #(T)->(1,T)->(N,T)
        pos_emb = pos_emb.unsqueeze(-1) #(N,T,1)
        dim_emb = torch.arange(0, self.TSAL_dim, 2).unsqueeze(0).unsqueeze(0).repeat(input_tensor.size()[0],time_step,1).to(self.device) #(input_dim/2)->(1,input_dim/2)->(N,T,input_dim/2)
        div_term = torch.exp(torch.arange(0, self.TSAL_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.TSAL_dim)).unsqueeze(0).unsqueeze(0).repeat(input_tensor.size()[0],time_step,1).to(self.device) #(input_dim/2)->(1,input_dim/2)->(N,T,input_dim/2)
        pos_emb = pos_emb.float()
        dim_emb = dim_emb.float()
        div_term = div_term.float()
        pe = torch.zeros(input_tensor.size()[0],time_step,self.TSAL_dim).to(self.device) #(N,T,input_dim)
        pe[:,:,0::2] = torch.sin(pos_emb * div_term) #(N,T,input_dim/2)
        pe[:,:,1::2] = torch.cos(pos_emb * div_term) #(N,T,input_dim/2)
        input_tensor = input_tensor + pe #(N,T,input_dim)
                
        Q_tensor = torch.matmul(input_tensor, self.TSAL_W_Q)  #(N,T,input_dim)->(N,T,input_dim)
        K_tensor = torch.matmul(input_tensor, self.TSAL_W_K)  #(N,T,input_dim)->(N,T,input_dim)
        V_tensor = torch.matmul(input_tensor, self.TSAL_W_V)  #(N,T,input_dim)->(N,T,input_dim)

        Q_tensor_ = torch.cat(torch.split(Q_tensor, int(self.TSAL_dim/self.head_num), 2), 0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)
        K_tensor_ = torch.cat(torch.split(K_tensor, int(self.TSAL_dim/self.head_num), 2), 0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)
        V_tensor_ = torch.cat(torch.split(V_tensor, int(self.TSAL_dim/self.head_num), 2), 0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)

        output_tensor = torch.matmul(Q_tensor_,K_tensor_.permute(0,2,1)) #(N*head_num,T,input_dim/head_num)->(N*head_num,T,T)
        output_tensor = output_tensor/(time_step ** 0.5)

        diag_val = torch.ones_like(output_tensor[0,:,:]).to(self.device) #(T,T)
        tril_tensor = torch.tril(diag_val).unsqueeze(0) #(T,T)->(1,T,T),where tril is lower_triangle matx.
        masks = tril_tensor.repeat(output_tensor.size()[0],1,1) #(1,T,T)->(N*head_num,T,T)
        padding = torch.ones_like(masks) * (-2 ** 32 + 1) 
        output_tensor = torch.where(masks.eq(0),padding,output_tensor) #(N*head_num,T,T),where replace lower_trianlge 0 with 1.
        output_tensor= F.softmax(output_tensor,1) 
        self.TSA_attn = output_tensor

        output_tensor = self.drop_layer(output_tensor) 
        N = output_tensor.size()[0]
        output_tensor = torch.matmul(output_tensor, V_tensor_) #(N*head_num,T,T),(N*head_num,T,input_dim/head_num)->(N*head_num,T,input_dim/head_num)
        output_tensor = torch.cat(torch.split(output_tensor,int(N/self.head_num),0),-1) #(N*head_num,T,input_dim/head_num) -> (N,T,input_dim)

        output_tensor = self.feedforward(output_tensor)
        output_tensor += input_tensor
        
        return output_tensor

    def CrossAttention_Layer(self, user_emb, item_emb):
        time_step = item_emb.size()[1]
        
        user_emb = user_emb.unsqueeze(1)
        
        Q_tensor = torch.matmul(user_emb, self.CAL_W_Q)  #(N,T,input_dim)->(N,T,input_dim)
        K_tensor = torch.matmul(item_emb, self.CAL_W_K)  #(N,T,input_dim)->(N,T,input_dim)
        V_tensor = torch.matmul(item_emb, self.CAL_W_V)  #(N,T,input_dim)->(N,T,input_dim)
        
        Q_tensor_ = torch.cat(torch.split(Q_tensor, int(self.CAL_dim/self.CAL_head_num), 2), 0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)
        K_tensor_ = torch.cat(torch.split(K_tensor, int(self.CAL_dim/self.CAL_head_num), 2), 0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)
        V_tensor_ = torch.cat(torch.split(V_tensor, int(self.CAL_dim/self.CAL_head_num), 2), 0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)

        output_tensor = torch.matmul(Q_tensor_,K_tensor_.permute(0,2,1)) #(N*head_num,T,input_dim/head_num)->(N*head_num,T,T)
        output_tensor = output_tensor/(time_step ** 0.5)

        diag_val = torch.ones_like(output_tensor[0,:,:]).to(self.device) #(T,T)
        tril_tensor = torch.tril(diag_val).unsqueeze(0) #(T,T)->(1,T,T),where tril is lower_triangle matx.
        masks = tril_tensor.repeat(output_tensor.size()[0],1,1) #(1,T,T)->(N*head_num,T,T)
        padding = torch.ones_like(masks) * (-2 ** 32 + 1) 
        output_tensor = torch.where(masks.eq(0),padding,output_tensor) #(N*head_num,T,T),where replace lower_trianlge 0 with 1.
        output_tensor= F.softmax(output_tensor,1) 
        self.CAL_attn = output_tensor

        output_tensor = self.CAL_drop_layer(output_tensor) 
        N = output_tensor.size()[0]
        output_tensor = torch.matmul(output_tensor, V_tensor_) #(N*head_num,T,T),(N*head_num,T,input_dim/head_num)->(N*head_num,T,input_dim/head_num)
        output_tensor = torch.cat(torch.split(output_tensor,int(N/self.CAL_head_num),0),-1) #(N*head_num,T,input_dim/head_num) -> (N,T,input_dim)

        # LayerNormed-Feed Forward + Residual Connection
        output_tensor = self.feedforward_cross(output_tensor)
        output_tensor += user_emb
        
        return output_tensor

    def forward(self,X_user_item,X_graph_base,for_pred=False):
        batch_users, batch_sequences, items_to_predict = X_user_item[0], X_user_item[1], X_user_item[2]
        edge_index, edge_type, node_no, short_term_part = X_graph_base[0], X_graph_base[1], X_graph_base[2], X_graph_base[3]
        x = self.node_embeddings(node_no)

        rate = torch.tensor([[1] for i in range(edge_type.size()[0])]).to(self.device)

        concat_states = []
        self.attn_weight_list = list()
        
        for conv in self.conv_modulelist:
            x = torch.tanh(conv(x, edge_index, edge_type, gate_emd2=None, rate=rate))
            concat_states.append(x)
            self.attn_weight_list.append(conv.attn_weight)
        
        for conv in self.short_conv_modulelist:
            for i in range(len(short_term_part)):
                short_edge_index,short_edge_type = short_term_part[i][0],short_term_part[i][1]
                x = torch.tanh(conv(x, short_edge_index, short_edge_type, gate_emd2=None, rate=rate))
            concat_states.append(x)
 
        concat_states = torch.cat(concat_states, 1)
        user_emb = concat_states[batch_users]
        item_embs_conv = concat_states[batch_sequences]
        item_embs = self.Temporal_Attention_Layer(item_embs_conv)
        item_embs = self.CrossAttention_Layer(user_emb, item_embs)
        
        '''
        user_emb : shape(bz,dim)
        item_embs : shape(bz,L,dim)
        items_to_predict(train) : shape(bz,2*H)
        items_to_predict(test) : shape(bz,topk)
        '''

        pe_w = self.predict_emb_w(items_to_predict) 
        pe_b = self.predict_emb_b(items_to_predict) 
        if for_pred:
            pe_w = pe_w.squeeze()
            pe_b = pe_b.squeeze()
            # user-pred_item
            res = user_emb.mm(pe_w.t()) + pe_b 
            # item-item 
            rel_score = torch.matmul(item_embs, pe_w.t().unsqueeze(0)) 
            rel_score = torch.sum(rel_score, dim=1) 
            res += rel_score  
            return res        
        else:
            # user-pred_item
            res = torch.baddbmm(pe_b, pe_w, user_emb.unsqueeze(2)).squeeze()
            # item-item 
            rel_score = item_embs.bmm(pe_w.permute(0, 2, 1))
            rel_score = torch.sum(rel_score, dim=1)
            res += rel_score
            return res,user_emb,item_embs ### Changed from item_embs_conv to item_embs