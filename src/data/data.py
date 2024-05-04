import pandas as pd
import numpy as np
import logging
from graph.graph import NeighborFinder
from utils.utils import RandEdgeSampler

class Data():
    """
    Data class for loading and splitting the dataset.

    Parameters:
        DATASET: str, the dataset name.
        args: argparse.ArgumentParser, the argument parser.        
    """
    def __init__(self, DATASET, args, user_id=None):
        ### Load data and train val test split
        g_df = pd.read_csv('/home/FYP/siddhant005/tagon/processed/ml_{}.csv'.format(DATASET))
        self.split_data(g_df, args, user_id)


    def split_data(self, g_df, args, user_id=None):
        """
        Split the data into training, validation, and test sets.

        Parameters:
            g_df: pd.DataFrame, the data frame of the dataset.
            args: argparse.ArgumentParser, the argument parser.
        """
        logger = logging.getLogger()
        
        
        if user_id is not None:
            g_df = g_df[g_df.u == user_id]
               
        val_time, test_time = list(np.quantile(g_df.ts, [0.80, 0.90]))
        
        src_l = g_df.u.values
        dst_l = g_df.i.values
        e_idx_l = g_df.idx.values
        label_l = g_df.label.values
        ts_l = g_df.ts.values
        
        max_src_index = src_l.max()
        self.max_idx = max(src_l.max(), dst_l.max())
        self.num_total_edges = len(src_l)

        total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
        num_total_unique_nodes = len(total_node_set)
        
        valid_train_flag = (ts_l <= val_time)
        
        self.train_src_l = src_l[valid_train_flag]
        self.train_dst_l = dst_l[valid_train_flag]
        self.train_ts_l = ts_l[valid_train_flag]
        self.train_e_idx_l = e_idx_l[valid_train_flag]
        self.train_label_l = label_l[valid_train_flag]
        self.valid_train_userset = set(np.unique(self.train_src_l))
        self.valid_train_itemset = set(np.unique(self.train_dst_l))
        logger.info(f'#interactions in train: {len(self.train_src_l)}')
        
        # select validation and test dataset
        valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
        valid_test_flag = ts_l > test_time
        
        # validation and test with all edges
        val_src_l = src_l[valid_val_flag]
        val_dst_l = dst_l[valid_val_flag]
        val_ts_l = ts_l[valid_val_flag]
        val_e_idx_l = e_idx_l[valid_val_flag]
        val_label_l = label_l[valid_val_flag]
        
        valid_is_old_node_edge = np.array([(a in self.valid_train_userset and b in self.valid_train_itemset) for a, b in zip(val_src_l, val_dst_l)])
        self.val_src_l = val_src_l[valid_is_old_node_edge]
        self.val_dst_l = val_dst_l[valid_is_old_node_edge]
        self.val_ts_l = val_ts_l[valid_is_old_node_edge]
        self.val_e_idx_l = val_e_idx_l[valid_is_old_node_edge]
        self.val_label_l = val_label_l[valid_is_old_node_edge]
        logger.info(f'#interactions in valid:{len(self.val_src_l)} ')

        test_src_l = src_l[valid_test_flag]
        test_dst_l = dst_l[valid_test_flag]
        test_ts_l = ts_l[valid_test_flag]
        test_e_idx_l = e_idx_l[valid_test_flag]
        test_label_l = label_l[valid_test_flag]
        
        test_is_old_node_edge = np.array([(a in self.valid_train_userset and b in self.valid_train_itemset) for a, b in zip(test_src_l, test_dst_l)])
        self.test_src_l = test_src_l[test_is_old_node_edge]
        self.test_dst_l = test_dst_l[test_is_old_node_edge]
        self.test_ts_l = test_ts_l[test_is_old_node_edge]
        self.test_e_idx_l = test_e_idx_l[test_is_old_node_edge]
        self.test_label_l = test_label_l[test_is_old_node_edge]
        logger.info(f'#interaction in test: {len(self.test_src_l)}')


        adj_list = [[] for _ in range(self.max_idx + 1)]
        for src, dst, eidx, ts in zip(self.train_src_l, self.train_dst_l, self.train_e_idx_l, self.train_ts_l):
            adj_list[src].append((dst, eidx, ts))
            adj_list[dst].append((src, eidx, ts))
        self.train_ngh_finder = NeighborFinder(adj_list, uniform=args.uniform)

        test_train_adj_list = [[] for _ in range(self.max_idx + 1)]
        for src, dst, eidx, ts in zip(self.train_src_l, self.train_dst_l, self.train_e_idx_l, self.train_ts_l):
            test_train_adj_list[src].append((dst, eidx, ts))
            test_train_adj_list[dst].append((src, eidx, ts))
        for src, dst, eidx, ts in zip(self.val_src_l, self.val_dst_l, self.val_e_idx_l, self.val_ts_l):
            test_train_adj_list[src].append((dst, eidx, ts))
            test_train_adj_list[dst].append((src, eidx, ts))
        self.test_train_ngh_finder = NeighborFinder(test_train_adj_list, uniform=args.uniform)


        # full graph with all the data for the test and validation purpose
        full_adj_list = [[] for _ in range(self.max_idx + 1)]
        for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
            full_adj_list[src].append((dst, eidx, ts))
            full_adj_list[dst].append((src, eidx, ts))
        self.full_ngh_finder = NeighborFinder(full_adj_list, uniform=args.uniform)
        
        self.train_rand_sampler = RandEdgeSampler(self.train_src_l, self.train_dst_l, self.train_ts_l)
        self.val_rand_sampler = RandEdgeSampler(src_l, dst_l, ts_l)
        self.test_rand_sampler = RandEdgeSampler(src_l, dst_l, ts_l)


    def set_test_data_for_one_user(self, user_id):
        """
        Set the test data for one user.

        Parameters:
            user_id: int, the user ID.
        """
        self.test_src_l = self.test_src_l[self.test_src_l == user_id]
        self.test_dst_l = self.test_dst_l[self.test_src_l == user_id]
        self.test_ts_l = self.test_ts_l[self.test_src_l == user_id]
        self.test_e_idx_l = self.test_e_idx_l[self.test_src_l == user_id]
        self.test_label_l = self.test_label_l[self.test_src_l == user_id]

        self.test_rand_sampler = RandEdgeSampler(self.test_src_l, self.test_dst_l, self.test_ts_l)
        