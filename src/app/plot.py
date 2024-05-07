import plotly.graph_objects as go
import networkx as nx
import json
import pandas as pd

def make_ctbg(user_hist, dataset, trim=10):
    user_hist = user_hist.sort_values(by='ts', ascending=False)
    user_hist = user_hist[:trim]
    csv_data = user_hist[['u', 'i', 'ts']]
    
    # convert ts to datetime DD-MM-YYYY HH:MM:SS
    csv_data['ts'] = pd.to_datetime(csv_data['ts'], unit='s')
    
    map_file_path = f'/Users/siddhantpathak/Desktop/Projects/tagon/processed/{dataset}_i_map.json'
    item_name_map = json.load(open(map_file_path))
    
    # Create a directed bipartite graph
    B = nx.DiGraph()

    # Add user and item nodes
    users = csv_data['u'].unique()
    items = csv_data['i'].unique()
    B.add_nodes_from(users, bipartite=0)  # Users
    B.add_nodes_from(items, bipartite=1)  # Items

    # Add edges from user nodes to item nodes
    for _, row in csv_data.iterrows():
        B.add_edge(row['u'], row['i'])

    # Generate positions for the nodes
    pos = nx.bipartite_layout(B, users)

    max_y = max(pos[node][1] for node in items)
    min_y = min(pos[node][1] for node in items)
    mid_y = (max_y + min_y) / 2
    for user in users:
        pos[user] = (pos[user][0], mid_y)  # Center the user node

    # Extract x and y coordinates for plotting
    edge_x = []
    edge_y = []
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
    node_colors = ['yellow' if node in users else 'blue' for node in B.nodes()]

    # Create the Plotly figure
    fig = go.Figure()

    # Add edges as lines
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=0.5, color='darkslategrey'),
                            hoverinfo='none'))

    # Add nodes as scatter points with hover information
    hover_texts = [
        f"User ID: {node}" if node in users else
        f"Item ID: {node}<br><i>Item Name:</i> {item_name_map.get(str(node), {}).get('title', 'Unknown Item')}<br><i>Timestamp:</i> {csv_data[csv_data['i'] == node]['ts'].iloc[0]}"
        for node in B.nodes()
    ]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers',
                            marker=dict(size=15, color=node_colors),
                            text=hover_texts,
                            hoverinfo='text',
                            hoverlabel=dict(bgcolor='white',
                                            bordercolor=node_colors,
                                            font=dict(color='black', family='Open Sans, sans-serif', size=13)),
                            showlegend=False))

    # Set layout properties
    fig.update_layout(
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False,
                                showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False,
                                showticklabels=False),
                    margin=dict(l=0, r=0, t=0, b=0),
                      annotations=arrows
                    ) 

    return fig

def make_recgraph(ctbg, rec_output):
    pass