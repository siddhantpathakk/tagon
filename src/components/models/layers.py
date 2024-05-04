import torch
# torch.cuda.empty_cache()
import torch.nn as nn

class MergeLayer(torch.nn.Module):
    """
    Two layer feedforward network for merging the input features.

    Parameters:
        dim1: int, the dimension of the first input.
        dim2: int, the dimension of the second input.
        dim3: int, the dimension of the first hidden layer.
        dim4: int, the dimension of the second hidden layer.

    Inputs:
        x1: torch.Tensor, the first input tensor with shape [N, L, dim1].
        x2: torch.Tensor, the second input tensor with shape [N, L, dim2].
    """
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(0.35)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):        
        x = torch.cat([x1, x2], dim=-1)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    


class MergeSelfAttnLayer(nn.Module):
    """
    MergeSelfAttnLayer is a class for merging and concatenating the input features.
    Meant for self-attention module.

    Parameters:
        feat_dim: int, the dimension of the first input.
        time_dim: int, the dimension of the second input.
        batch_size: int, the batch size.
        n: int, the number of heads.

    Inputs:
        seq: torch.Tensor, the first input tensor with shape [N, L, feat_dim].
        seq_t: torch.Tensor, the second input tensor with shape [N, L, time_dim].
    """
    def __init__(self, feat_dim, time_dim, batch_size, n):
        super(MergeSelfAttnLayer, self).__init__()
        
        self.dropout = nn.Dropout(0.35)
        self.layer_norm = nn.LayerNorm(feat_dim+time_dim+feat_dim)
        self.fc1 = nn.Linear(feat_dim+time_dim+feat_dim, batch_size * n)
        self.fc2 = nn.Linear(batch_size * n, feat_dim+time_dim)
        self.act = nn.ReLU()            
        
    def forward(self, seq, seq_t):
        x = torch.cat([seq, seq_t], dim=-1)
        bs, n, _ = x.size()
        x = x.view(bs * n, -1)
        
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.view(bs, n, -1)
        return x
        