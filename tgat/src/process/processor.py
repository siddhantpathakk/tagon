class DataProcessor:
    def __init__(self, path_name, meta_path, out_df, out_feat, out_node_feat, u_map_file, i_map_file):
        self.PATH = path_name
        self.meta_path = meta_path
        self.OUT_DF = out_df
        self.OUT_FEAT = out_feat
        self.OUT_NODE_FEAT = out_node_feat
        self.u_map_file = u_map_file
        self.i_map_file = i_map_file
        
    
    def preprocess(self):
        raise NotImplementedError("preprocess method not implemented")
    
    def reindex(self):
        raise NotImplementedError("reindex method not implemented")
    
    def run(self):
        raise NotImplementedError("run method not implemented")
    