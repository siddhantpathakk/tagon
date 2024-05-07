import plotly.graph_objects as go
import networkx as nx
import json
import pandas as pd


def make_ctbg(user_hist, dataset, trim=10):
    # Sort and trim the history
    user_hist = user_hist.sort_values(by='ts', ascending=False).head(trim)
    user_hist['ts'] = pd.to_datetime(user_hist['ts'], unit='s')

    print(user_hist[['u', 'i', 'ts']])

    # Load the item mapping file
    map_file_path = f'/Users/siddhantpathak/Desktop/Projects/tagon/processed/{dataset}_i_map.json'
    item_name_map = json.load(open(map_file_path))

    # ignore
    item_name_df = pd.DataFrame(item_name_map).T
    print(item_name_df)

    # Create a directed bipartite graph
    B = nx.DiGraph()
    B.add_nodes_from(user_hist['u'].unique(), bipartite=0)
    B.add_nodes_from(user_hist['i'].unique(), bipartite=1)
    edges = user_hist.apply(lambda row: B.add_edge(row['u'], row['i']), axis=1)

    # Generate positions for the nodes
    pos = nx.bipartite_layout(B, user_hist['u'].unique())

    # Sorting the items by timestamp for positioning
    sorted_items = user_hist.sort_values(by='ts', ascending=True)['i'].unique()
    y_pos = {node: i for i, node in enumerate(sorted_items)}
    for node, position in pos.items():
        if node in y_pos:
            pos[node] = (position[0], y_pos[node])

    # Center the user node by adjusting the y-position
    item_y_positions = [pos[node][1] for node in sorted_items]
    if item_y_positions:
        mid_y = (max(item_y_positions) + min(item_y_positions)) / 2
    else:
        mid_y = 0

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

        arrows.append(dict(
            x=x1, y=y1, xref='x', yref='y',
            ax=x0, ay=y0, axref='x', ayref='y',
            showarrow=True, arrowhead=1, arrowsize=0.8, arrowwidth=1, arrowcolor='darkslategrey'))

    node_x = [pos[node][0] for node in B.nodes()]
    node_y = [pos[node][1] for node in B.nodes()]
    node_colors = ['white' if node in user_hist['u'].unique()
                   else '#FFD700' for node in B.nodes()]

    hover_texts = [
        f"User ID: {node}" if node in user_hist['u'].unique() else
        f"<b>Timestamp</b>: {user_hist[user_hist['i'] == node]['ts'].iloc[0]}<br><br><b>Item ID</b>: {node}<br><b>Item Name</b>: {get_item_name(item_name_df, node)[0]}<br><b>Year</b>: {get_item_name(item_name_df, node)[1]}"
        for node in B.nodes()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(
        width=0.5, color='darkslategrey'), hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=18, color=node_colors),
                             text=hover_texts, hoverinfo='text', textposition='bottom left',
                             hoverlabel=dict(bgcolor='white', bordercolor=node_colors,
                                                                                 font=dict(color='black', family='Open Sans, sans-serif', size=14)))
    )

    fig.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), margin=dict(l=0, r=0, t=0, b=0), annotations=arrows)

    return fig

def get_item_name(item_df, item_id):
    name = item_df[item_df['item_id'] == item_id]['title'].iloc[0] if item_id in item_df['item_id'].values else 'UNKNOWN ITEM'
    if name == 'UNKNOWN ITEM':
        return name, "???"
    else:
        name = name.split("(")
        item_name, year_str = name[0].strip(), name[1].replace(")", "").strip()
        return item_name, year_str

def make_recgraph(ctbg, rec_output):
    pass