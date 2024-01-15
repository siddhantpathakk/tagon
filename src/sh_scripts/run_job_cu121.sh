#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --mem=15G
#SBATCH --output=/home/FYP/siddhant005/fyp/src/out/logs/outputs_cuda121.out
#SBATCH --error=/home/FYP/siddhant005/fyp/src/out/logs/errors_cuda121.err
#SBATCH --job-name=fyp

module load anaconda
module load cuda/12.1
source activate retagnn_pyg
/home/FYP/siddhant005/.conda/envs/retagnn_pyg/bin/python /home/FYP/siddhant005/fyp/src/main.py 