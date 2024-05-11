
import ast
import pandas as pd

def create_row(row):
    return {
        'reviewerID': row['reviewerID'],
        'asin': row['asin'],
        'reviewerName': row['reviewerName'] if 'reviewerName' in row else '',
        'summary': row['summary'] if 'summary' in row else ''
    }

def create_user_row(row):
    return {
        'reviewerID': row['reviewerID'],
        'reviewerName': row['reviewerName'] if 'reviewerName' in row else ''
    
    }

def fix_review_user_item_mapping(dataset_name):
    file_path = f'reviews_{dataset_name}_5.json'
    with open(file_path, 'r') as f:
        data = f.readlines()
        rows = []
        for row in data:
            row = ast.literal_eval(row)
            rows.append(create_row(row))
            
    df = pd.DataFrame(rows)
    print(df.head(1))
    df.to_csv(f'review_meta/review_{dataset_name}.csv', index=False)

def create_user_mapping(dataset_name):
    file_path = f'reviews_{dataset_name}_5.json'
    with open(file_path, 'r') as f:
        data = f.readlines()
        rows = []
        for row in data:
            row = ast.literal_eval(row)
            rows.append(create_user_row(row))
            
    df = pd.DataFrame(rows)
    print(df.head(2))
    df.to_csv(f'user_meta/user_{dataset_name}.csv', index=False)
    
def main(dataset_name):
    print(f"Fixing {dataset_name} dataset")
    fix_review_user_item_mapping(dataset_name)
    create_user_mapping(dataset_name)
    
main('Baby')
main('Digital_Music')
main('Toys_and_Games')
main('Tools_and_Home_Improvement')
