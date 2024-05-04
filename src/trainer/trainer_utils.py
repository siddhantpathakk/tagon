import torch
from models.TAGON import TAGON


def setup_model(data, args, n_nodes, GPU, NUM_LAYER, USE_TIME, AGG_METHOD, ATTN_MODE, SEQ_LEN, NUM_HEADS, DROP_OUT, NODE_DIM, TIME_DIM, load_pretrain=None):
    device = torch.device('cuda:{}'.format(GPU)) if torch.cuda.is_available() else torch.device('cpu')
    
    model = TAGON(data.train_ngh_finder, n_nodes+1, args,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)

    if load_pretrain:
        checkpoint = torch.load(load_pretrain, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    return model

def setup_optimizer(model, LEARNING_RATE, load_pretrain=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if load_pretrain:
        checkpoint = torch.load(load_pretrain, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return optimizer

def bpr_loss(pos_score, neg_score):
    loss = -((pos_score - neg_score).sigmoid().log().mean())
    return loss
