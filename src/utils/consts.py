import time

datetime_str = time.strftime("%Y%m%d-%H%M%S")

RANK_RESULTS_DIR = f"./tmp/rank_results"
# RANK_RESULTS_FILE = lambda args: f"{RANK_RESULTS_DIR}/{args.data}_TAGON_{datetime_str}"
SAVE_MODEL_DIR = lambda args: f"./tmp/{args.data}/"

pretrain_path = lambda args: f"./tmp/{args.data}/{args.data}_TAGON.pt"