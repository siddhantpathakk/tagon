SAVE_MODEL_DIR = lambda args, cwd: cwd + f"/tmp/{args.data}/"

pretrain_path = lambda args, cwd: cwd + f"/tmp/ckpts/slab/{args.data}/{args.data}_TARGON.pt"
# pretrain_path = lambda args: f"./tmp/ckpts/scse_gpu/{args.data}/{args.data}_TAGON.pt"

user_history_path = lambda args, cwd, userID: cwd + f"/tmp/{args.data}_UID{userID}_history.csv"
infer_output_path = lambda args, cwd, userID: cwd + f"/tmp/{args.data}_UID{userID}_output.csv"
