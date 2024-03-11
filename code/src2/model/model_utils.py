import torch
from torch import nn
from model.FFN import PointWiseFeedForward
from model.improvedGNN import LongTermGNN, ShortTermGNN
from model.attention_layers import TemporalSequentialAttentionLayer_v2, CrossAttentionLayer_v2

def build_model_variant(args, relation_num):
    """
    Possible model variants:
    1. Long term gnn + short term gnn + TSAL+CAL + pointwise FFN
    2. Long term gnn + short term gnn + TSAL+CAL + NN
    3. Long term gnn + short term gnn + TSAL + Pointwise
    4. Long term gnn + short term gnn + TSAL + NN
    
    returns:
        long_term_gnn, short_term_gnn, temporal_seq_attn_layer, cross_attn_layer
    """
    
    model_variant = args.model_variant
    device = args.device
    attn_dimension = (args.conv_layer_num + args.short_conv_layer_num) * args.dim
        
    
    if model_variant == 1:
        FFN = PointWiseFeedForward(attn_dimension, args.attn_drop, device=device).to(device)
        long_term_gnn = LongTermGNN(dim=args.dim, 
                                     conv_layer_num=args.conv_layer_num, 
                                     relation_num=relation_num,
                                     num_bases=args.num_bases, 
                                     device=device).to(device)
        short_term_gnn = ShortTermGNN(dim=args.dim, 
                                     conv_layer_num=args.conv_layer_num, 
                                     relation_num=relation_num,
                                     num_bases=args.num_bases, 
                                     device=device).to(device)
        temporal_seq_attn_layer = TemporalSequentialAttentionLayer_v2(attn_dim=attn_dimension,
                                                                      num_heads=args.TSAL_head_num,
                                                                      feedforward=FFN,
                                                                      dropout=args.attn_drop,
                                                                      device=device).to(device)
        cross_attn_layer = CrossAttentionLayer_v2(attn_dim=attn_dimension,
                                                  num_heads=args.CAL_head_num,
                                                  feedforward=FFN,
                                                  dropout=args.attn_drop,
                                                  device=device).to(device)
    
    elif model_variant == 2:
        FFN = nn.Sequential(
                nn.Linear(attn_dimension, attn_dimension).to(device), 
                torch.nn.LayerNorm(attn_dimension, eps=1e-8).to(device),
            ).to(device)
        long_term_gnn = LongTermGNN(dim=args.dim, 
                                     conv_layer_num=args.conv_layer_num, 
                                     relation_num=relation_num,
                                     num_bases=args.num_bases, 
                                     device=device).to(device)
        short_term_gnn = ShortTermGNN(dim=args.dim, 
                                     conv_layer_num=args.conv_layer_num, 
                                     relation_num=relation_num,
                                     num_bases=args.num_bases, 
                                     device=device).to(device)
        temporal_seq_attn_layer = TemporalSequentialAttentionLayer_v2(attn_dim=attn_dimension,
                                                                      num_heads=args.TSAL_head_num,
                                                                      feedforward=FFN,
                                                                      dropout=args.attn_drop,
                                                                      device=device).to(device)
        cross_attn_layer = CrossAttentionLayer_v2(attn_dim=attn_dimension,
                                                  num_heads=args.CAL_head_num,
                                                  feedforward=FFN,
                                                  dropout=args.attn_drop,
                                                  device=device).to(device)
    
    elif model_variant == 3:
        FFN = PointWiseFeedForward(attn_dimension, args.attn_drop, device=device).to(device)
        long_term_gnn = LongTermGNN(dim=args.dim, 
                                     conv_layer_num=args.conv_layer_num, 
                                     relation_num=relation_num,
                                     num_bases=args.num_bases, 
                                     device=device).to(device)
        short_term_gnn = ShortTermGNN(dim=args.dim, 
                                     conv_layer_num=args.conv_layer_num, 
                                     relation_num=relation_num,
                                     num_bases=args.num_bases, 
                                     device=device).to(device)
        temporal_seq_attn_layer = TemporalSequentialAttentionLayer_v2(attn_dim=attn_dimension,
                                                                      num_heads=args.TSAL_head_num,
                                                                      feedforward=FFN,
                                                                      dropout=args.attn_drop,
                                                                      device=device).to(device)
        cross_attn_layer = None
    
    elif model_variant == 4:
        FFN = nn.Sequential(
                nn.Linear(attn_dimension, attn_dimension).to(device), 
                torch.nn.LayerNorm(attn_dimension, eps=1e-8).to(device),
            ).to(device)
        long_term_gnn = LongTermGNN(dim=args.dim, 
                                     conv_layer_num=args.conv_layer_num, 
                                     relation_num=relation_num,
                                     num_bases=args.num_bases, 
                                     device=device).to(device)
        short_term_gnn = ShortTermGNN(dim=args.dim, 
                                     conv_layer_num=args.conv_layer_num, 
                                     relation_num=relation_num,
                                     num_bases=args.num_bases, 
                                     device=device).to(device)
        temporal_seq_attn_layer = TemporalSequentialAttentionLayer_v2(attn_dim=attn_dimension,
                                                                      num_heads=args.TSAL_head_num,
                                                                      feedforward=FFN,
                                                                      dropout=args.attn_drop,
                                                                      device=device).to(device)
        cross_attn_layer = None
    
    else:
        raise ValueError(f"Invalid model variant: {model_variant}")
    
    return long_term_gnn, short_term_gnn, temporal_seq_attn_layer, cross_attn_layer
