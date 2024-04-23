#!/bin/sh
source ~/.bashrc

python src/train.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python src/test.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000

python src/train.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python src/test.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000

python src/train.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python src/test.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000

python src/train.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python src/test.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000

python src/train.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python src/test.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
