#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --mem=15G
#SBATCH --output=/home/FYP/siddhant005/fyp/experiments/reta-gnn/src/logs/%x_%j.out
#SBATCH --error=/home/FYP/siddhant005/fyp/experiments/reta-gnn/src/logs/err/%x_%j.err
#SBATCH --job-name=retaML1M

module load anaconda
module load cuda/12.1
source activate retagnn_pyg
home/FYP/siddhant005/.conda/envs/retagnn_pyg/bin/python3 /home/FYP/siddhant005/fyp/experiments/reta-gnn/src/main.py --file_path /home/FYP/siddhant005/fyp/experiments/reta-gnn/data/processed/ml-1m/