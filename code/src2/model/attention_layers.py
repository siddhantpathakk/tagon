import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class BaseAttentionLayer(nn.Module):
    def __init__(self, attn_dim, num_heads, feedforward, dropout=0.1, device='cuda'):
        super(BaseAttentionLayer, self).__init__()
        
        self.attn_dim = attn_dim
        self.head_num = num_heads
        self.device = device
        
        self.W_Q = Variable(torch.zeros(self.attn_dim, self.attn_dim).type(torch.FloatTensor), requires_grad=True).to(device)
        self.W_K = Variable(torch.zeros(self.attn_dim, self.attn_dim).type(torch.FloatTensor), requires_grad=True).to(device)
        self.W_V = Variable(torch.zeros(self.attn_dim, self.attn_dim).type(torch.FloatTensor), requires_grad=True).to(device)

        self.drop_layer = nn.Dropout(p=dropout)
        self.feedforward = feedforward.to(device)
        
        self.attn = None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.W_Q = nn.init.xavier_uniform_(self.W_Q)
        self.W_K = nn.init.xavier_uniform_(self.W_K)
        self.W_V = nn.init.xavier_uniform_(self.W_V) 

    def forward(self):
        raise NotImplementedError("forward method is not implemented")
    
    def positional_encoding(self):
        raise NotImplementedError("positional_encoding method is not implemented")


class TemporalSequentialAttentionLayer_v2(BaseAttentionLayer):
    def __init__(self, attn_dim, num_heads, feedforward, dropout=0.1, device='cuda'):
        super(TemporalSequentialAttentionLayer_v2, self).__init__(attn_dim, num_heads, feedforward, dropout, device)
        self.reset_parameters()
    
    def forward(self, input_tensor):
        time_step = input_tensor.size()[1]
        
        input_tensor = input_tensor + self.positional_encoding(input_tensor=input_tensor)
        
        Q_tensor = torch.matmul(input_tensor, self.W_Q)  #(N,T,input_dim)->(N,T,input_dim)
        K_tensor = torch.matmul(input_tensor, self.W_K) 
        V_tensor = torch.matmul(input_tensor, self.W_V)  
        
        Q_tensor_ = torch.cat(torch.split(Q_tensor, int(self.attn_dim/self.head_num), 2), 0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)
        K_tensor_ = torch.cat(torch.split(K_tensor, int(self.attn_dim/self.head_num), 2), 0)   
        V_tensor_ = torch.cat(torch.split(V_tensor, int(self.attn_dim/self.head_num), 2), 0)   
        
        output_tensor = torch.matmul(Q_tensor_,K_tensor_.permute(0,2,1)) #(N*head_num,T,input_dim/head_num)->(N*head_num,T,T)
        output_tensor = output_tensor/(time_step ** 0.5)

        diag_val = torch.ones_like(output_tensor[0,:,:]).to(self.device)
        tril_tensor = torch.tril(diag_val).unsqueeze(0)
        masks = tril_tensor.repeat(output_tensor.size()[0],1,1) 
        padding = torch.ones_like(masks) * (-2 ** 32 + 1) 
        output_tensor = torch.where(masks.eq(0),padding,output_tensor) 
        output_tensor= F.softmax(output_tensor,1) 
        self.attn = output_tensor

        output_tensor = self.drop_layer(output_tensor) 
        N = output_tensor.size()[0]
        output_tensor = torch.matmul(output_tensor, V_tensor_) 
        output_tensor = torch.cat(torch.split(output_tensor,int(N/self.head_num),0),-1)

        output_tensor = self.feedforward(output_tensor)
        output_tensor += input_tensor
        
        return output_tensor

        
    def positional_encoding(self, input_tensor):
        # positional embedding using sin and cos function
        time_step = input_tensor.size()[1]
        
        pos_emb = torch.arange(0, time_step).unsqueeze(0).repeat(input_tensor.size()[0],1).to(self.device) 
        pos_emb = pos_emb.unsqueeze(-1) 
        
        dim_emb = torch.arange(0, self.attn_dim, 2).unsqueeze(0).unsqueeze(0).repeat(input_tensor.size()[0],time_step,1).to(self.device) 
        div_term = torch.exp(torch.arange(0, self.attn_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.attn_dim)).unsqueeze(0).unsqueeze(0).repeat(input_tensor.size()[0],time_step,1).to(self.device)
        
        pos_emb = pos_emb.float()
        dim_emb = dim_emb.float()
        div_term = div_term.float()
        
        pe = torch.zeros(input_tensor.size()[0],time_step,self.attn_dim).to(self.device) 
        pe[:,:,0::2] = torch.sin(pos_emb * div_term) 
        pe[:,:,1::2] = torch.cos(pos_emb * div_term)
        
        return pe #(N,T,input_dim)
        

class CrossAttentionLayer_v2(BaseAttentionLayer):
    def __init__(self, attn_dim, num_heads, feedforward, dropout=0.1, device='cuda'):
        super(CrossAttentionLayer_v2, self).__init__(attn_dim, num_heads, feedforward, dropout, device)
        self.reset_parameters()

    def forward(self, user_emb, item_emb):
        time_step = item_emb.size()[1]
        
        user_emb = user_emb.unsqueeze(1)
        
        Q_tensor = torch.matmul(user_emb, self.W_Q)  #(N,T,input_dim)->(N,T,input_dim)
        K_tensor = torch.matmul(item_emb, self.W_K)  
        V_tensor = torch.matmul(item_emb, self.W_V)  
        
        Q_tensor_ = torch.cat(torch.split(Q_tensor, int(self.attn_dim/self.head_num), 2), 0)   #(N,T,input_dim)->(N*head_num,T,input_dim/head_num)
        K_tensor_ = torch.cat(torch.split(K_tensor, int(self.attn_dim/self.head_num), 2), 0)   
        V_tensor_ = torch.cat(torch.split(V_tensor, int(self.attn_dim/self.head_num), 2), 0)   

        output_tensor = torch.matmul(Q_tensor_, K_tensor_.permute(0,2,1)) #(N*head_num,T,input_dim/head_num)->(N*head_num,T,T)
        output_tensor = output_tensor/(time_step ** 0.5)

        diag_val = torch.ones_like(output_tensor[0,:,:]).to(self.device) 
        tril_tensor = torch.tril(diag_val).unsqueeze(0) 
        masks = tril_tensor.repeat(output_tensor.size()[0],1,1)
        padding = torch.ones_like(masks) * (-2 ** 32 + 1) 
        output_tensor = torch.where(masks.eq(0),padding,output_tensor) 
        output_tensor= F.softmax(output_tensor,1) 
        self.attn = output_tensor

        output_tensor = self.drop_layer(output_tensor) 
        N = output_tensor.size()[0]
        output_tensor = torch.matmul(output_tensor, V_tensor_) 
        output_tensor = torch.cat(torch.split(output_tensor,int(N/self.head_num),0),-1) 

        output_tensor = self.feedforward(output_tensor)
        output_tensor += user_emb # Why not item_emb?
        
        return output_tensor