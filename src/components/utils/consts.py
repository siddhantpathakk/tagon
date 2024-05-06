SAVE_MODEL_DIR = lambda args, cwd: cwd + f"/tmp/{args.data}/"

pretrain_path_slab = lambda args, cwd: cwd + f"/tmp/ckpts/slab/{args.data}/{args.data}_TARGON.pt"
pretrain_path_scse = lambda args, cwd: cwd + f"/tmp/ckpts/scse_gpu/{args.data}/{args.data}_TAGON.pt"

pretrain_app_path_slab = lambda args, cwd: cwd + f"/components/tmp/ckpts/slab/{args.data}/{args.data}_TARGON.pt"
pretrain_app_path_scse = lambda args, cwd: cwd + f"/components/tmp/ckpts/scse_gpu/{args.data}/{args.data}_TAGON.pt"

user_history_path = lambda args, cwd, userID: cwd + f"/tmp/{args.data}_UID{userID}_history.csv"
infer_output_path = lambda args, cwd, userID: cwd + f"/tmp/{args.data}_UID{userID}_output.csv"
