import plotly.graph_objects as go
import networkx as nx
import json
import pandas as pd
import datetime

def make_ctbg(user_hist, dataset, trim=10, theme='light'):
    user_hist = user_hist.sort_values(by='ts', ascending=False)
    user_hist['ts'] = pd.to_datetime(user_hist['ts'], unit='s')
    user_hist = user_hist[:trim]

    B = nx.DiGraph()
    B.add_nodes_from(user_hist['u'].unique(), bipartite=0)
    B.add_nodes_from(user_hist['i'].unique(), bipartite=1)
    user_hist.apply(lambda row: B.add_edge(row['u'], row['i']), axis=1)

    pos = nx.bipartite_layout(B, user_hist['u'].unique())

    sorted_items = user_hist.sort_values(by='ts', ascending=False)['i'].unique()
    y_pos = {node: i for i, node in enumerate(sorted_items)}
    for node, position in pos.items():
        if node in y_pos:
            pos[node] = (position[0], y_pos[node])

    item_labels = {item: f"i<sub>{index + 1}</sub>" for index,
                   item in enumerate(sorted_items)}
    user_labels = {user: f"u<sub>{index + 1}</sub>" for index,
                   user in enumerate(user_hist['u'].unique())}

    item_y_positions = [pos[node][1] for node in sorted_items]
    mid_y = (max(item_y_positions) + min(item_y_positions)) / 2 if item_y_positions else 0
    for user in user_hist['u'].unique():
        pos[user] = (pos[user][0], mid_y)

    edge_x, edge_y = [], []
    arrows = []
    for edge in B.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    item_node_color = "#e7d1ff" if theme.startswith('#FF') else "#FFFF00"

    node_x = [pos[node][0] for node in B.nodes()]
    node_y = [pos[node][1] for node in B.nodes()]
    node_colors = ['#A4D4B4' if node in user_hist['u'].unique()  # else '#E4D9FF' for node in B.nodes ()]
                   else '#FFFF00' for node in B.nodes()]
    
    for i in range(2, len(node_colors)):
        node_colors[i] = generate_lighter_shade_of_color(node_colors[i - 1], factor=0.35)

    node_texts = [user_labels.get(node, item_labels.get(node, ""))
                  for node in B.nodes()]

    hover_texts = []
    user_id = None
    for node in B.nodes():
        if node in user_hist['u'].unique():
            if dataset == 'ml-100k':
                hover_texts.append(f"<b>Graph Node ID:</b> {node}")
            else:
                hover_texts.append(f"""<b>Graph Node ID:</b> {node}<br><b>Username:</b> {get_username(node, dataset)[0]}<br><b>User ID:</b> {get_username(node, dataset)[1]}""")
            user_id = node
        else:
            if dataset == 'ml-100k':
                hover_texts.append(f"""<b>ID:</b> {node}<br><b>Movie Name:</b> {get_item_name(node, dataset)}<br><b>Interacted on:</b> {user_hist[user_hist['i'] == node]['ts'].iloc[0].strftime('%d-%B-%Y %I:%M%p')}""")
            else:
                hover_texts.append(
                    f"""<b>Graph Node ID:</b> {node}<br><b>ASIN:</b> {str(get_asin(node, dataset)).strip()}<br><b>Item Name:</b> {str(get_item_name(node, dataset)).strip()}<br><b>Interacted on:</b> {user_hist[user_hist['i'] == node]['ts'].iloc[0].strftime('%d-%B-%Y %I:%M%p')}<br><b>Review given:</b> "{str(get_review(node, user_id, dataset)).strip()}"<br><b>Categories:</b> {str(get_categories(node, dataset)).strip()}""")
            

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(
        width=0.5, color='#e50914'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_texts, 
                             marker=dict(size=30, color=node_colors),
                             hoverinfo='text', hovertext=hover_texts, textposition='middle center',
                             hoverlabel=dict(bgcolor='white',
                                             bordercolor=node_colors,
                                             font=dict(
                                                    color='black', family='Times New Roman', size=14),
                                                    align="left"),
                             textfont=dict(size=16, family='Open Sans, sans-serif', color='black')))

    fig.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), margin=dict(l=0, r=0, t=10, b=10),
                      annotations=arrows)

    return fig

def generate_lighter_shade_of_color(color, factor=0.5):
    color = color.lstrip('#')
    rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    new_rgb = tuple(int((255 - val) * factor + val) for val in rgb)
    return f"#{''.join([hex(val)[2:].zfill(2) for val in new_rgb])}"

def get_item_name(item_id, dataset='amzn'):
    if dataset == 'ml-100k':
        item_df = pd.read_json(f'/Users/siddhantpathak/Desktop/Projects/tagon/processed/ml-100k_i_map.json').T
        name = item_df[item_df['item_id'] == item_id]['title'].iloc[0] if item_id in item_df['item_id'].values else 'UNKNOWN ITEM'
    else:
        asin = get_asin(item_id, dataset)
        name = get_name_from_asin(asin, dataset)
    return name

def get_name_from_asin(asin, dataset_name):
    item_df = pd.read_csv(f'/Users/siddhantpathak/Desktop/Projects/tagon/datasets/item_meta/meta_{dataset_name}.csv')
    name = item_df[item_df['asin'] == asin]['title'].iloc[0]
    return name if name else ''

def get_asin(item_id, dataset_name):
    df = pd.read_csv(f'/Users/siddhantpathak/Desktop/Projects/tagon/processed/merged_ml_{dataset_name}.csv')
    return df[df['i'] == item_id]['asin'].iloc[0]

def get_review(item_id, user_id, dataset_name):
    df = pd.read_csv(f'/Users/siddhantpathak/Desktop/Projects/tagon/datasets/review_meta/review_{dataset_name}.csv')
    asin = get_asin(item_id, dataset_name)
    revID = get_reviewerID(user_id, dataset_name)
    return df[(df['asin'] == asin) & (df['reviewerID'] == revID)]['summary'].values[0]


def get_categories(node_id, dataset_name):
    import ast
    asin = get_asin(node_id, dataset_name)
    df = pd.read_csv(f'/Users/siddhantpathak/Desktop/Projects/tagon/datasets/item_meta/meta_{dataset_name}.csv')
    cats = ast.literal_eval(df[(df['asin'] == asin)]['categories'].values[0])
    return cats[:1] if len(cats) > 1 else cats

def get_reviewerID(user_id, dataset_name):
    df = pd.read_csv(
        f'/Users/siddhantpathak/Desktop/Projects/tagon/processed/merged_ml_{dataset_name}.csv')
    return df[df['u'] == user_id]['reviewerID'].iloc[0]

def get_username(user_id, dataset_name):
    reviewer_id = get_reviewerID(user_id, dataset_name)
    df = pd.read_csv(f'/Users/siddhantpathak/Desktop/Projects/tagon/datasets/user_meta/user_{dataset_name}.csv')
    return df[df['reviewerID'] == reviewer_id]['reviewerName'].iloc[0], reviewer_id
    
    
    
def make_recgraph(ctbg, rec_output):
    return ctbg


def make_plotly_table(df, title_):
    # Enhanced Plotly table with readability improvements
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='navy',  # Darker color for header
            align='left',
            # White text on navy background, larger font
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left',
            # Slightly larger font for cells
            font=dict(color='black', size=11),
            height=30  # Taller cells for better readability
        )
    )],
        layout=go.Layout(
        title=go.layout.Title(text=title_),
        title_font=dict(size=16)  # Larger title font
    ))
    return fig
