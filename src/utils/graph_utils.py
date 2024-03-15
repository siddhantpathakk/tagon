import numpy as np
import pandas as pd

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, ts_list):
        self.edges = {}
        for u, i in zip(src_list, dst_list):
            if u in self.edges:
                self.edges[u].add(i)
            else:
                self.edges[u] = set([i])
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        df = pd.DataFrame({'u': src_list, 'i': dst_list, 'ts': ts_list})

        self.items_popularity = df.groupby('i')['i'].count().to_dict()
        self.time_sorted_df = df.sort_values(by=['ts'])

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def sample_neg(self, src_list):
        dst_set = set(self.dst_list)
        neg_dst = np.zeros(len(src_list), dtype=int)
        for ind, u in enumerate(src_list):
            random_neg = np.random.choice(list(dst_set-self.edges[u]), 1)[0]
            neg_dst[ind] = random_neg
        return neg_dst

    def popularity_based_sample_neg(self, src_list):
        dst_set = set(self.dst_list)
        neg_dst = np.zeros(len(src_list), dtype=int)
        for ind, u in enumerate(src_list):
            neg_candidate = list(dst_set - self.edges[u])
            neg_candidate_pop_items = []
            neg_candidate_pop = []
            for neg_item in neg_candidate:
                neg_candidate_pop_items.append(neg_item)
                neg_candidate_pop.append(self.items_popularity[neg_item])

            total_popularity = sum(neg_candidate_pop)
            neg_candidate_pop_prob = [pop / total_popularity for pop in neg_candidate_pop]

            random_neg = np.random.choice(neg_candidate_pop_items, size=1, p=neg_candidate_pop_prob)
            neg_dst[ind] = random_neg
        return neg_dst

    def timelypopularity_based_sample_neg(self, src_list, ts_list):
        neg_dst = np.zeros(len(src_list), dtype=int)
        range_ind = 1500
        all_ts = self.time_sorted_df['ts'].values
        for ind, u in enumerate(src_list):
            ts_cut = ts_list[ind]
            min_ts_diff_ind = np.argmin(abs(all_ts - ts_cut))
            prev_min_ind = max(0, min_ts_diff_ind - range_ind)
            later_max_ind = min(self.time_sorted_df.shape[0], min_ts_diff_ind + range_ind)

            timely_selected_df = self.time_sorted_df.iloc[prev_min_ind:later_max_ind, :]

            selected_items_popularity = timely_selected_df.groupby('i')['i'].count().to_dict()
            neg_candidate = list(set(list(selected_items_popularity.keys())) - self.edges[u])
            if len(neg_candidate) == 0:
                print(selected_items_popularity)
                print(self.edges[u])
                print(prev_min_ind, later_max_ind)
            neg_candidate_pop_items = []
            neg_candidate_pop = []
            for neg_item in neg_candidate:
                neg_candidate_pop_items.append(neg_item)
                neg_candidate_pop.append(selected_items_popularity[neg_item])
            
            total_popularity = sum(neg_candidate_pop)
            neg_candidate_pop_prob = [pop / total_popularity for pop in neg_candidate_pop]

            random_neg = np.random.choice(neg_candidate_pop_items, size=1, p=neg_candidate_pop_prob)
            neg_dst[ind] = random_neg

        return neg_dst



class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
       
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l
        
        self.off_set_l = off_set_l
        
        self.uniform = uniform
        
    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])
           
            
            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, off_set_l
        
    def find_before(self, src_idx, cut_time):
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_idx

        left = 0
        right = len(neighbors_idx) - 1
        
        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid
                
        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i,  num_neighbors - len(ngh_eidx):] = ngh_eidx
                    
        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for _ in range(k -1):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1] # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_est, ngn_t_est, num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, num_neighbors) # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records

