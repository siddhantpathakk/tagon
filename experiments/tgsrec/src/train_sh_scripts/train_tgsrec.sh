#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --mem=15G
#SBATCH --output=/home/FYP/siddhant005/fyp/experiments/tgsrec/%x_%j.out 
#SBATCH --error=/home/FYP/siddhant005/fyp/experiments/tgsrec/%x_%j.err 
#SBATCH --job-name=tgsrec

module load anaconda
source activate tgsrec
/home/FYP/siddhant005/.conda/envs/tgsrec/bin/python3 /home/FYP/siddhant005/fyp/experiments/tgsrec/src/run_TGREC.py -d ml-100k --uniform --bs 600 --lr 0.001 --n_degree 30 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --n_layer 2 --prefix Video_Games_bce --node_dim 32 --time_dim 32 --drop_out 0.3 --reg 0.3 --negsampleeval 1000