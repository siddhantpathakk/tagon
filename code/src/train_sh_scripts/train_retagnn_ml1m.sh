#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=q_ug48
#SBATCH --mem=20G
#SBATCH --output=/home/FYP/siddhant005/fyp/code/src/logs/%x_%j.out
#SBATCH --error=/home/FYP/siddhant005/fyp/code/src/logs/err/%x_%j.err
#SBATCH --job-name=ML1M

module load anaconda
module load cuda/12.1
source activate retagnn_pyg
/home/FYP/siddhant005/.conda/envs/retagnn_pyg/bin/python3 /home/FYP/siddhant005/fyp/code/src/main.py --dataset ml1m --verbose 1 --debug 0