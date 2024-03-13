import argparse

from process.process_reddit import RedditProcessor
from process.process_wikipedia import WikipediaProcessor
from process.process_ml100k import MovielensProcessor

def get_opt():
    
    dataset_options = ['reddit', 'wikipedia', 'movielens-100k']
    
    parser = argparse.ArgumentParser(description='Preprocess data for TGAT')
    parser.add_argument('--dataset', type=str, default='ml100k', choices=dataset_options, help='Dataset to use')
    
    return parser.parse_args()


data_directories = {
    'reddit': {
        'path_name': './data/raw/reddit/reddit.csv', # 'reddit.csv' is a file in the 'raw' directory
        'meta_path': None,
        'out_df': './data/processed/reddit/reddit.csv',
        'out_feat': './data/processed/reddit/ml_reddit.npy',
        'out_node_feat': './data/processed/reddit/ml_reddit_node.npy',
        'u_map_file': None,
        'i_map_file': None
    },
    
    'wikipedia': {
        'path_name': './data/raw/wikipedia/wikipedia.csv',
        'meta_path': None,
        'out_df': './data/processed/wikipedia/wikipedia.csv',
        'out_feat': './data/processed/wikipedia/ml_wikipedia.npy',
        'out_node_feat': './data/processed/wikipedia/ml_wikipedia_node.npy',
        'u_map_file': None,
        'i_map_file': None
    },
    
    'movielens-100k': {
        'path_name': './data/raw/ml100k/u.data',
        'meta_path': './data/raw/ml100k/u.item',
        'out_df': './data/processed/ml100k/ml_100k.csv',
        'out_feat': None,
        'out_node_feat': None,
        'u_map_file': './data/processed/ml100k/u_map.npy',
        'i_map_file': './data/processed/ml100k/i_map.npy'
    },
}

if __name__ == '__main__':
    dataset = get_opt().dataset
    
    if dataset == 'reddit':
        processor = RedditProcessor(**data_directories[dataset])
        processor.run()
        
    elif dataset == 'wikipedia':
        processor = WikipediaProcessor(**data_directories[dataset])
        processor.run()
        
    elif dataset == 'movielens-100k':
        processor = MovielensProcessor(**data_directories[dataset])
        processor.run()

        
    else:
        raise ValueError('Dataset not supported')
     