#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --mem=32G
#SBATCH --output=./tmp/SLURM_%x_%j.out
#SBATCH --error=./tmp/SLURM_%x_%j.err
#SBATCH --job-name=ML100K

module load anaconda
module load cuda/12.1
source ~/.bashrc
source activate retagnn_pyg
which python


python src/main.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python src/main.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --test_mode 
python src/main.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --infer_mode

python src/main.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 
python src/main.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --test_mode
python src/main.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --infer_mode

python src/main.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python src/main.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --test_mode
python src/main.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --infer_mode

python src/main.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python src/main.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --test_mode
python src/main.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --infer_mode

python src/main.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python src/main.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --test_mode
python src/main.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000 --infer_mode