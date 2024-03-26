#!/bin/sh
#SBATCH --job-name=val_Toy
#SBATCH --output=log/eval/slurm/%x_%j.out
#SBATCH --error=log/eval/slurm/%x_%j.err
#SBATCH --partition=rtx3090_slab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

source ~/.bashrc
source ~/miniconda3/bin/activate siddhant_env
which python
python TGSRec/val_TGREC.py -d Toys_and_Games --uniform --bs 128 --lr 0.001 --n_degree 30 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --n_layer 2 --prefix amzn_music --node_dim 32 --time_dim 32 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --model TGSREC