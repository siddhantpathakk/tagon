import os

def get_model_save_path(args):
    MODEL_SAVE_PATH = f'/home/FYP/siddhant005/fyp/log/saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
    get_checkpoint_path = lambda epoch: f'/home/FYP/siddhant005/fyp/log/saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'
    SAVE_MODEL_DIR = f"./log/saved_models/{args.data}"
    
    if not os.path.isdir(SAVE_MODEL_DIR):
        os.mkdir(SAVE_MODEL_DIR)
        
    SAVE_MODEL_PATH = SAVE_MODEL_DIR + f"/checkpoint.{args.bs}_{args.n_degree}_{args.n_epoch}_{args.n_head}_{args.drop_out}_{args.time}_{args.n_layer}_{args.n_degree}_{args.node_dim}_{args.time_dim}_{args.lr}.pth.tar"


    return SAVE_MODEL_DIR, SAVE_MODEL_PATH, MODEL_SAVE_PATH, get_checkpoint_path

def get_rank_results_paths(args):
    RANK_RESULTS_DIR = f"/home/FYP/siddhant005/fyp/log/rank_results/{args.data}"
    
    if not os.path.isdir(RANK_RESULTS_DIR):
        os.mkdir(RANK_RESULTS_DIR)
        
    RANK_RESULTS_FILE = RANK_RESULTS_DIR + f"/{args.bs}_{args.n_degree}_{args.n_epoch}_{args.n_head}_{args.drop_out}_{args.time}_{args.n_layer}_{args.n_degree}_{args.node_dim}_{args.time_dim}_{args.lr}"
    
    return RANK_RESULTS_FILE, RANK_RESULTS_DIR
