#!/bin/sh
#SBATCH --job-name=ML
#SBATCH --output=log/test/slurm/out/%x_%j.out
#SBATCH --error=log/test/slurm/%x_%j.err
#SBATCH --partition=rtx3090_slab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

source ~/.bashrc
source ~/miniconda3/bin/activate siddhant_env
which python
python TGSRec/run_TGREC.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000