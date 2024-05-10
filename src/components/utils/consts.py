SAVE_MODEL_DIR = lambda args, cwd: cwd + f"/tmp/{args.data}/"

pretrain_path_slab = lambda args, cwd: cwd + f"/tmp/ckpts/slab/{args.data}/{args.data}_TARGON.pt"
pretrain_path_scse = lambda args, cwd: cwd + f"/tmp/ckpts/scse_gpu/{args.data}/{args.data}_TAGON.pt"

pretrain_app_path_slab = lambda args, cwd: cwd + f"/components/tmp/ckpts/slab/{args.data}/{args.data}_TARGON.pt"
pretrain_app_path_scse = lambda args, cwd: cwd + f"/components/tmp/ckpts/scse_gpu/{args.data}/{args.data}_TAGON.pt"
pretrain_app_path_new = lambda args, cwd: cwd + f"/components/tmp/LATEST/{args.data}-5_TAGON.pt"

pretrain_ml100k_slab = lambda args, cwd: cwd + f"/components/tmp/ckpts/slab/ml-100k/ml-100k_TARGON.pt"
pretrain_ml100k_scse = lambda args, cwd: cwd + f"/components/tmp/ckpts/scse_gpu/ml-100k/ml-100k_TAGON.pt"

user_history_path = lambda args, cwd, userID: cwd + f"/tmp/{args.data}_UID{userID}_history.csv"
infer_output_path = lambda args, cwd, userID: cwd + f"/tmp/{args.data}_UID{userID}_output.csv"
