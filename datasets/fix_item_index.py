import pandas as pd
i_map=[]
i_set = set()
ind = 0
with open('review_meta/review_Baby.json', 'r') as f:
    for line in f:
        line = eval(line)
        if line['asin'] not in i_set:
            row = {'asin': line['asin'], 'index': ind}
            ind += 1
            i_map.append(row)
            i_set.add(line['asin'])
            
print(pd.DataFrame(i_map))