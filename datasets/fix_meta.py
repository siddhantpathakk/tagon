import pandas as pd
import json
import ast

def create_row(row):
    if 'categories' not in row:
        row['categories'] = ''
    if 'title' not in row:
        row['title'] = ''
    return {
        'asin': row['asin'],
        'title': row['title'],
        'categories': row['categories'],
    }

# Path to your JSON file
def fix_asin_meta_mapping(dataset_name):
    file_path = f'meta_{dataset_name}.json'
    rows = []
    # read the file line by line, seperate using comma and convert to a json object
    with open(file_path, 'r') as f:
        data = f.readlines()
        for row in data:
            row = ast.literal_eval(row)
            rows.append(create_row(row))
        
    print(len(rows))
    df = pd.DataFrame(rows)
    print(df.head())    
    df.to_csv(f'item_meta/meta_{dataset_name}.csv', index=False)
    
    
fix_asin_meta_mapping('Baby')
fix_asin_meta_mapping('Digital_Music')
fix_asin_meta_mapping('Toys_and_Games')
fix_asin_meta_mapping('Tools_and_Home_Improvement')