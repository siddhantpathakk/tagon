#!/bin/bash
#SBATCH --partition=SCSEGPU_UG 
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --job-name=ultraML1M
#SBATCH --output=/home/FYP/siddhant005/fyp/experiments/ultra-gcn/logs/%x_%j.out 
#SBATCH --error=/home/FYP/siddhant005/fyp/experiments/ultra-gcn/logs/%x_%j.err

module load anaconda
source activate ultragcn
/home/FYP/siddhant005/.conda/envs/ultragcn/bin/python3 /home/FYP/siddhant005/fyp/experiments/ultra-gcn/src/main.py --config_file /home/FYP/siddhant005/fyp/experiments/ultra-gcn/src/config/ultragcn_movielens1m_m1.ini --dataset movielens