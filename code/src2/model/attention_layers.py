import torch
import torch.nn as nn

class TemporalSequentialAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward, dropout=0.1):
        super(TemporalSequentialAttentionLayer, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.feedforward = feedforward

    def forward(self, input_tensor):
        time_step = input_tensor.size(1)
        pos_emb = self.positional_encoding(
            time_step, self.embed_dim, input_tensor.device)
        input_tensor = input_tensor + pos_emb

        attn_output, _ = self.multihead_attn(input_tensor, input_tensor, input_tensor)
        attn_output = self.dropout(attn_output)
        output_tensor = self.feedforward(attn_output)
        output_tensor = self.dropout(output_tensor)
        output_tensor = self.layer_norm(output_tensor + input_tensor)

        return output_tensor

    def positional_encoding(self, length, dim, device):
        position = torch.arange(
            length, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, dim, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / dim)).to(device)
        pos_emb = torch.zeros(length, dim).to(device)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        return pos_emb.unsqueeze(0)


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.feedforward = feedforward


    def forward(self, user_emb, item_emb):
        user_emb = user_emb.unsqueeze(1)  # Add sequence length dimension
        attn_output, _ = self.multihead_attn(user_emb, item_emb, item_emb)
        attn_output = self.dropout(attn_output)
        output_tensor = self.feedforward(attn_output)
        output_tensor = self.dropout(output_tensor)
        output_tensor = self.layer_norm(output_tensor + user_emb)
        
        return output_tensor
