#!/bin/bash

##Job Script for FYP

#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=q_ug24
#SBATCH --mem=32G
#SBATCH --job-name=TGSRec
#SBATCH --output=/home/FYP/siddhant005/fyp/train_sh_scripts/slurm_logs/train/output_%x_%j.out
#SBATCH --error=/home/FYP/siddhant005/fyp/train_sh_scripts/slurm_logs/train/error_%x_%j.err

module load anaconda
module load cuda/12.1
source activate retagnn_pyg
which python
python /home/FYP/siddhant005/fyp/src/main.py -d ml-100k --uniform