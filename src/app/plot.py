import plotly.graph_objects as go
import networkx as nx
import json
import pandas as pd


def make_ctbg(user_hist, dataset, trim=10):
    # Sort and trim the history
    user_hist = user_hist.sort_values(by='ts', ascending=False)
    user_hist['ts'] = pd.to_datetime(user_hist['ts'], unit='s')
    user_hist = user_hist[:trim]

    # Load the item mapping file
    map_file_path = f'/Users/siddhantpathak/Desktop/Projects/tagon/processed/{dataset}_i_map.json'
    item_name_map = json.load(open(map_file_path))
    item_name_df = pd.DataFrame(item_name_map).T
    print(item_name_df)
    
    # Create a directed bipartite graph
    B = nx.DiGraph()
    B.add_nodes_from(user_hist['u'].unique(), bipartite=0)
    B.add_nodes_from(user_hist['i'].unique(), bipartite=1)
    user_hist.apply(lambda row: B.add_edge(row['u'], row['i']), axis=1)

    # Generate positions for the nodes
    pos = nx.bipartite_layout(B, user_hist['u'].unique())

    # Sorting the items by timestamp for positioning
    sorted_items = user_hist.sort_values(by='ts', ascending=False)['i'].unique()
    y_pos = {node: i for i, node in enumerate(sorted_items)}
    for node, position in pos.items():
        if node in y_pos:
            pos[node] = (position[0], y_pos[node])

    # Assign labels to item nodes with subscript formatting
    item_labels = {item: f"i<sub>{index + 1}</sub>" for index,
                   item in enumerate(sorted_items)}
    user_labels = {user: f"u<sub>{index + 1}</sub>" for index,
                   user in enumerate(user_hist['u'].unique())}

    # Adjust the y-position of the user nodes to be centered
    item_y_positions = [pos[node][1] for node in sorted_items]
    mid_y = (max(item_y_positions) + min(item_y_positions)) / 2 if item_y_positions else 0
    for user in user_hist['u'].unique():
        pos[user] = (pos[user][0], mid_y)

    # Extract x and y coordinates for plotting
    edge_x, edge_y = [], []
    arrows = []
    for edge in B.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in B.nodes()]
    node_y = [pos[node][1] for node in B.nodes()]
    node_colors = ['#A4D4B4' if node in user_hist['u'].unique()  # else '#E4D9FF' for node in B.nodes ()]
                   else '#FFFF00' for node in B.nodes()]
    
    # for all node colors after the second one, make them dimmer than the previous one
    for i in range(2, len(node_colors)):
        node_colors[i] = generate_lighter_shade_of_color(node_colors[i - 1], factor=0.25)
    print(node_colors)

    # Node labels for item and user nodes with subscript
    node_texts = [user_labels.get(node, item_labels.get(node, ""))
                  for node in B.nodes()]

    # Hover texts including item and user details
    hover_texts = [
        f"User ID: {node}" if node in user_hist['u'].unique() else
        f"<i>#{node}</i><br>{get_item_name(item_name_df, node)}<br>{user_hist[user_hist['i'] == node]['ts'].iloc[0]}"
        for node in B.nodes()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(
        width=0.5, color='#e50914'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_texts, marker=dict(size=30, color=node_colors),
                             hoverinfo='text', hovertext=hover_texts, textposition='middle center',
                             hoverlabel=dict(bgcolor='white',
                                             bordercolor=node_colors,
                                             font=dict(color='black', family='Open Sans, sans-serif', size=14)),
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

def get_item_name(item_df, item_id):
    name = item_df[item_df['item_id'] == item_id]['title'].iloc[0] if item_id in item_df['item_id'].values else 'UNKNOWN ITEM'
    return name

def make_recgraph(ctbg, rec_output):
    return ctbg