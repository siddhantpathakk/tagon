#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --mem=15G
#SBATCH --output=/home/FYP/siddhant005/fyp/experiments/tisasrec/%x_%j.out 
#SBATCH --error=/home/FYP/siddhant005/fyp/experiments/tisasrec/%x_%j.err 
#SBATCH --job-name=tisasrec

module load anaconda
source activate tisasrec
python3 main.py --dataset=ml-1m --train_dir=default --device=cuda