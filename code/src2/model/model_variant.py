from torch import nn
from model.FFN import PointWiseFeedForward
from model.improvedGNN import LongTermGNN, ShortTermGNN
from model.attention_layers import TemporalSequentialAttentionLayer_v2, CrossAttentionLayer_v2

def build_model_variant(model_variant, config, relation_num):
    """
    Model variants
        Long term gnn + short term gnn + TSAL+CAL + pointwise FFN
        Long term gnn + short term gnn + TSAL+CAL + NN
        Long term gnn + short term gnn + TSAL + Pointwise
        Long term gnn + short term gnn + TSAL + NN

    Returns
        long_term_gnn: LongTermGNN
        short_term_gnn: ShortTermGNN
        temporal_seq_attn_layer: TemporalSequentialAttentionLayer_v2
        cross_attn_layer: CrossAttentionLayer_v2
    """
    attn_dimension = (config.conv_layer_num + config.short_conv_layer_num) * config.dim
    
    long_term_gnn = LongTermGNN(dim=config.dim, 
                                conv_layer_num=config.conv_layer_num, 
                                relation_num=relation_num,
                                num_bases=config.num_bases, 
                                device=config.device)
    
    short_term_gnn = ShortTermGNN(dim=config.dim, 
                                    conv_layer_num=config.short_conv_layer_num, 
                                    relation_num=relation_num,
                                    num_bases=config.num_bases, 
                                    device=config.device)
        
    nn_FFN = nn.Sequential(
            nn.Linear(attn_dimension, attn_dimension).to(config.device),
            nn.LayerNorm(attn_dimension, eps=1e-8).to(config.device)
        ).to(config.device)
    
    pointwise_FFN = nn.Sequential(
            PointWiseFeedForward(attn_dimension, config.attn_drop, device=config.device).to(config.device),
            nn.LayerNorm(attn_dimension, eps=1e-8).to(config.device)
        ).to(config.device)
    
    if model_variant==1:    
        temporal_seq_attn_layer = TemporalSequentialAttentionLayer_v2(attn_dim=attn_dimension,
                                                                      num_heads=config.TSAL_head_num,
                                                                      feedforward=pointwise_FFN,
                                                                      dropout=config.attn_drop,
                                                                      device=config.device)
        
        cross_attn_layer = CrossAttentionLayer_v2(attn_dim=attn_dimension,
                                                  num_heads=config.CAL_head_num,
                                                  feedforward=pointwise_FFN,
                                                  dropout=config.attn_drop,
                                                  device=config.device)
        
    elif model_variant==2:
        
        temporal_seq_attn_layer = TemporalSequentialAttentionLayer_v2(attn_dim=attn_dimension,                                                              
                                                                        num_heads=config.TSAL_head_num,
                                                                        feedforward=nn_FFN,
                                                                        dropout=config.attn_drop,
                                                                        device=config.device)
        
        cross_attn_layer = CrossAttentionLayer_v2(attn_dim=attn_dimension,
                                                    num_heads=config.CAL_head_num,
                                                    feedforward=nn_FFN,
                                                    dropout=config.attn_drop,
                                                    device=config.device)
        
    elif model_variant==3:
        temporal_seq_attn_layer = TemporalSequentialAttentionLayer_v2(attn_dim=attn_dimension,
                                                                        num_heads=config.TSAL_head_num,
                                                                        feedforward=pointwise_FFN,
                                                                        dropout=config.attn_drop,
                                                                        device=config.device)
        
        cross_attn_layer = None
        
    elif model_variant==4:

        temporal_seq_attn_layer = TemporalSequentialAttentionLayer_v2(attn_dim=attn_dimension,
                                                                        num_heads=config.TSAL_head_num,
                                                                        feedforward=nn_FFN,
                                                                        dropout=config.attn_drop,
                                                                        device=config.device)
        
        cross_attn_layer = None
        
    return long_term_gnn, short_term_gnn, temporal_seq_attn_layer, cross_attn_layer