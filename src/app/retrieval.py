import pandas as pd

def get_item_name(item_id, dataset='amzn'):
    if dataset == 'ml-100k':
        item_df = pd.read_json(
            f'/Users/siddhantpathak/Desktop/Projects/tagon/processed/ml-100k_i_map.json').T
        name = item_df[item_df['item_id'] ==
                       item_id]['title'].iloc[0] if item_id in item_df['item_id'].values else 'UNKNOWN ITEM'
    else:
        asin = get_asin(item_id, dataset)
        name = get_name_from_asin(asin, dataset)
    return name


def get_name_from_asin(asin, dataset_name):
    item_df = pd.read_csv(
        f'/Users/siddhantpathak/Desktop/Projects/tagon/datasets/item_meta/meta_{dataset_name}.csv')
    name = item_df[item_df['asin'] == asin]['title'].iloc[0]
    return name if name else ''


def get_asin(item_id, dataset_name):
    df = pd.read_csv(
        f'/Users/siddhantpathak/Desktop/Projects/tagon/processed/merged_ml_{dataset_name}.csv')
    return df[df['i'] == item_id]['asin'].iloc[0]


def get_review(item_id, user_id, dataset_name):
    df = pd.read_csv(
        f'/Users/siddhantpathak/Desktop/Projects/tagon/datasets/review_meta/review_{dataset_name}.csv')
    asin = get_asin(item_id, dataset_name)
    revID = get_reviewerID(user_id, dataset_name)
    return df[(df['asin'] == asin) & (df['reviewerID'] == revID)]['summary'].values[0]


def get_categories(node_id, dataset_name):
    import ast
    asin = get_asin(node_id, dataset_name)
    df = pd.read_csv(
        f'/Users/siddhantpathak/Desktop/Projects/tagon/datasets/item_meta/meta_{dataset_name}.csv')
    cats = ast.literal_eval(df[(df['asin'] == asin)]['categories'].values[0])
    return cats[:1] if len(cats) > 1 else cats


def get_reviewerID(user_id, dataset_name):
    df = pd.read_csv(
        f'/Users/siddhantpathak/Desktop/Projects/tagon/processed/merged_ml_{dataset_name}.csv')
    return df[df['u'] == user_id]['reviewerID'].iloc[0]


def get_username(user_id, dataset_name):
    reviewer_id = get_reviewerID(user_id, dataset_name)
    df = pd.read_csv(
        f'/Users/siddhantpathak/Desktop/Projects/tagon/datasets/user_meta/user_{dataset_name}.csv')
    return df[df['reviewerID'] == reviewer_id]['reviewerName'].iloc[0], reviewer_id


def prepare_recs_for_graph(recs, reference_point, n_recs):
    rec = recs[recs['i'] == reference_point]['predicted'].iloc[0][:n_recs]
    timestamp = recs[recs['i'] == reference_point]['ts'].iloc[0]
    user = recs[recs['i'] == reference_point]['u'].iloc[0]
    rec_dict = {
        'u': [user] * n_recs,
        'i': rec,
        'ts': [timestamp] * n_recs
    }
    return pd.DataFrame(rec_dict)