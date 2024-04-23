#!/bin/sh

source ~/.bashrc
source ~/miniconda3/bin/activate siddhant_env
which python
python TGSRec/run_TGREC_ablation.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 16 --time_dim 16 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000

python TGSRec/run_TGREC_ablation.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 0 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 0 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 0 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 0 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 0 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000

python TGSRec/run_TGREC_ablation.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn  --time pos --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn  --time pos --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn  --time pos --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn  --time pos --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method attn  --time pos --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000

python TGSRec/run_TGREC_ablation.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 10 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 10 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 10 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 10 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 10 --agg_method attn --attn_mode map --gpu 0 --n_head 2 --n_layer 2 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000

python TGSRec/run_TGREC_ablation.py -d Digital_Music --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method lstm --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d ml-100k --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method lstm --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Tools_and_Home_Improvement --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method lstm --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Toys_and_Games --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method lstm --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000
python TGSRec/run_TGREC_ablation.py -d Baby --uniform --bs 150 --lr 0.01 --n_degree 20 --agg_method lstm --attn_mode map --gpu 0 --n_head 2 --n_layer 1 --node_dim 8 --time_dim 8 --drop_out 0.3 --reg 0.3 --negsampleeval 1000