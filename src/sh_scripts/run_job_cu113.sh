#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --mem=10G
#SBATCH --output=/home/FYP/siddhant005/fyp/out/logs/outputs_cuda113.out
#SBATCH --error=/home/FYP/siddhant005/fyp/out/logs/errors_cuda113.err
#SBATCH --job-name=fyp

module load anaconda
source activate retagnn
python3 -V
python3 train.py