import numpy as np
import pandas as pd
from process.processor import DataProcessor

class WikipediaProcessor(DataProcessor):
    def __init__(self,path_name, meta_path, out_df, out_feat, out_node_feat, u_map_file, i_map_file):
        super().__init__(path_name, meta_path, out_df, out_feat, out_node_feat, u_map_file, i_map_file)
        
    def preprocess(self):
        u_list, i_list, ts_list, label_list = [], [], [], []
        feat_l = []
        idx_list = []
        
        with open(self.PATH) as f:
            s = next(f)
            print(s)
            for idx, line in enumerate(f):
                e = line.strip().split(',')
                u = int(e[0])
                i = int(e[1])
                
                
                
                ts = float(e[2])
                label = int(e[3])
                
                feat = np.array([float(x) for x in e[4:]])
                
                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                label_list.append(label)
                idx_list.append(idx)
                
                feat_l.append(feat)
        return pd.DataFrame({'u': u_list, 
                            'i':i_list, 
                            'ts':ts_list, 
                            'label':label_list, 
                            'idx':idx_list}), np.array(feat_l)
        
    def reindex(self, df):
        assert(df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert(df.i.max() - df.i.min() + 1 == len(df.i.unique()))
        
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u
        
        new_df = df.copy()
        print(new_df.u.max())
        print(new_df.i.max())
        
        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
        
        print(new_df.u.max())
        print(new_df.i.max())
        
        return new_df
    
    def run(self):
        df, feat = self.preprocess()
        new_df = self.reindex(df)
        
        print(feat.shape)
        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        feat = np.vstack([empty, feat])
        
        max_idx = max(new_df.u.max(), new_df.i.max())
        rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
        
        print(feat.shape)
        new_df.to_csv(self.OUT_DF)
        np.save(self.OUT_FEAT, feat)
        np.save(self.OUT_NODE_FEAT, rand_feat)
    
