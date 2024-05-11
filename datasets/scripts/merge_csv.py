import pandas as pd
import json

datasets = [
    # "Digital_Music",
    # "Baby",
    # "Toys_and_Games",
    # "Tools_and_Home_Improvement"
    "ml-100k"
]

root = f'/Users/siddhantpathak/Desktop/Projects/tagon/datasets/'

for dataset_name in datasets:
    print(f"Processing {dataset_name} dataset")
    data_ml_csv = pd.read_csv(root + f'{dataset_name}/data/ml_{dataset_name}.csv')

    print(data_ml_csv.tail(3))
    print(data_ml_csv.i.max(), data_ml_csv.i.min())

    i_map = pd.read_json(root + f'{dataset_name}/map/{dataset_name}_i_map.json').T.reset_index().rename(columns={'index': 'i'})
    print(i_map.tail(3))

    u_map_dict = json.load(open(root + f'{dataset_name}/map/{dataset_name}_u_map.json', 'r'))
    u_map = pd.DataFrame.from_dict(u_map_dict, orient='index').reset_index().rename(columns={'index': 'u'})
    u_map['u'] = u_map['u'].astype(int)
    print(u_map.tail(3))

    data_ml_csv = data_ml_csv.merge(i_map, on='i', how='left')
    print(data_ml_csv.tail(3))

    data_ml_csv = data_ml_csv.merge(u_map, on='u', how='left')
    data_ml_csv.columns = ['u', 'i', 'ts', 'label', 'idx', 'asin', 'title', 'reviewerID']
    print(data_ml_csv.tail(3))

    data_ml_csv.to_csv(root + f'{dataset_name}/data/merged_ml_{dataset_name}.csv', index=False)