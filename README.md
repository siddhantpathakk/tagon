# Graph Neural Networks for Sequential Recommendation

### Things to try
1. Try swapping order of TSAL and CAL


### Dataset variants
1. Movielens100K raw
2. Movielens100K processed (subtracted)
3. Movielens100K processed2 (scaled divide)
4. Movielens1M raw
5. Movielens1M processed (subtracted)
6. Movielens1M processed2 (scaled divide)


### Model variants
1. Long term gnn + short term gnn + TSAL+CAL + pointwise FFN
2. Long term gnn + short term gnn + TSAL+CAL + NN
3. Long term gnn + short term gnn + TSAL + Pointwise
4. Long term gnn + short term gnn + TSAL + NN


### Optimizer variants
1. Adam
2. AdamW


### Hyperparameter tuning options:
1. Domain bases = L, H, hop
2. Model based = l2 (1e-3  3e-3  1e-4)


### Possible baselines
1. Bert4Rec
2. LightGCN
3. Reta GNN
4. CoPE
5. GRU4Rec
6. TGSRec


### Demonstration ideas:
1. Streamlit app: user enters the site, searches from the database of items (movies), chooses what he likes (at least 20), and then recommend new ones. Show how those movies are recommended, using graphs (pyviz)
