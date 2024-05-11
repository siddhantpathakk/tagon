import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from retrieval import get_item_name, get_asin, get_review, get_categories, get_username, prepare_recs_for_graph

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
                   else item_node_color for node in B.nodes()]
    

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

    
def make_rec_hist(user_history, rec_output, reference_point, n_recs=5):
    rec = prepare_recs_for_graph(rec_output, reference_point, n_recs)
    new_user_hist = user_history.copy()
    new_user_hist = pd.concat([new_user_hist, rec])
    new_user_hist = new_user_hist.sort_values(by='ts', ascending=False)
    return new_user_hist



def make_ctbg_with_recommendations(user_hist, dataset, trim=10, theme='light', reference_id=None, recommendations=None):
    # Preparing the historical and recommendation data
    user_hist = user_hist.sort_values(by='ts', ascending=False)
    user_hist['ts'] = pd.to_datetime(user_hist['ts'], unit='s')
    user_hist = user_hist[:trim]

    # Graph construction
    B = nx.DiGraph()
    user_nodes = user_hist['u'].unique()
    item_nodes = user_hist['i'].unique()

    # Add user and item nodes
    B.add_nodes_from(user_nodes, bipartite=0)
    B.add_nodes_from(item_nodes, bipartite=1)

    # Connect user to historical items
    user_hist.apply(lambda row: B.add_edge(row['u'], row['i']), axis=1)

    # Node positioning
    pos = nx.bipartite_layout(B, user_nodes)

    # center user node wrt 
    sorted_items = user_hist.sort_values(by='ts', ascending=False)['i'].unique()
    item_y_positions = [pos[node][1] for node in sorted_items]
    mid_y = (max(item_y_positions) + min(item_y_positions)) / 2 if item_y_positions else 0
    for user in user_hist['u'].unique():
        pos[user] = (pos[user][0], mid_y)
        
    # Adding recommended items and their edges
    if recommendations is not None and reference_id is not None:
        for item_id in recommendations:
            if item_id not in B:
                B.add_node(item_id, bipartite=1)
            B.add_edge(reference_id, item_id)

        # Adjust positions for recommendation items near the reference point, take the reference point as the center
        ref_pos = pos[reference_id]
        for item_id in recommendations:
            pos[item_id] = (ref_pos[0] + 1, ref_pos[1] + (recommendations.index(item_id) - len(recommendations) / 2) * 0.2)
            
            

    # Prepare to draw the graph
    edge_x, edge_y = [], []
    for edge in B.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Node colors and labels
    node_x = [pos[node][0] for node in B.nodes()]
    node_y = [pos[node][1] for node in B.nodes()]

    # node_counter = 0
    # node_colors = []
    # for node in B.nodes():
    #     if node in user_nodes:
    #         node_colors.append('#A4D4B4')
    #     if node in item_nodes:
    #         base_color = '#FFD700' if node_counter == 0 else node_colors[node_counter - 1]
    #         new_node_color = generate_lighter_shade_of_color(base_color, factor=0.05)
    #         node_colors.append(new_node_color)
    #         node_counter += 1
    #     if node in recommendations:
    #         node_colors.append('#FF0000')

    node_colors = ['#A4D4B4' if node in user_nodes else '#FFFF00' if node in item_nodes else '#FF0000' for node in B.nodes()]

    print(node_colors)


    node_texts = []
    for index, node in enumerate(B.nodes()):
        if node in user_nodes:
            node_texts.append(f"user")
        elif node in recommendations:
            ref_index = list(B.nodes()).index(reference_id) + 1
            order = recommendations.index(node) + 1
            node_texts.append(f"i<sub>{ref_index},{order}</sub>")
        else:
            node_texts.append(f"item")


    # More informative hover texts
    hover_texts = []
    user_id = None
    for node in B.nodes():
        if node in user_hist['u'].unique():
            # User node hover text
            if dataset == 'ml-100k':
                hover_texts.append(f"<b>Graph Node ID:</b> {node}")
            else:
                hover_texts.append(
                    f"""<b>Graph Node ID:</b> {node}<br><b>Username:</b> {get_username(node, dataset)[0]}<br><b>User ID:</b> {get_username(node, dataset)[1]}""")
            user_id = node
        elif node in recommendations and dataset != 'ml-100k':
            # Recommendation node hover text
            hover_texts.append(f"<b>Item ID:</b> {node}<br><b>Item Name:</b> {get_item_name(node, dataset)}<br><b>Categories:</b> {get_categories(node, dataset)}")
        elif node in recommendations and dataset == 'ml-100k':
            # Recommendation node hover text
            hover_texts.append(
                f"<b>Item ID:</b> {node}<br><b>Item Name:</b> {get_item_name(node, dataset)}")
        else:
            # Historical item node hover text
            if dataset == 'ml-100k':
                hover_texts.append(
                    f"""<b>ID:</b> {node}<br><b>Movie Name:</b> {get_item_name(node, dataset)}<br><b>Interacted on:</b> {user_hist[user_hist['i'] == node]['ts'].iloc[0].strftime('%d-%B-%Y %I:%M%p')}""")
            else:
                hover_texts.append(
                    f"""<b>Graph Node ID:</b> {node}<br><b>ASIN:</b> {get_asin(node, dataset)}<br><b>Item Name:</b> {get_item_name(node, dataset)}<br><b>Interacted on:</b> {user_hist[user_hist['i'] == node]['ts'].iloc[0].strftime('%d-%B-%Y %I:%M%p')}<br><b>Review given:</b> "{get_review(node, user_id, dataset)}"<br><b>Categories:</b> {get_categories(node, dataset)}""")

    fig = go.Figure(layout=dict(
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False),
        autosize=True,
        hovermode='closest'
    ))
    
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                  line=dict(width=0.5, color='#e50914'), hoverinfo='none'))
    
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_texts,
                             marker=dict(size=30, color=node_colors),
                             hoverinfo='text', hovertext=hover_texts, textposition='middle center',
                                hoverlabel=dict(bgcolor='white',
                                                bordercolor=node_colors,
                                                font=dict(
                                                    color='black', family='Times New Roman', size=14), align="left"),
                                textfont=dict(size=10, family='Open Sans, sans-serif', color='black')))

    fig.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), margin=dict(l=0, r=0, t=10, b=10))
    
    annotation_text = f'<b>Reference Point:</b> i<sub>{list(B.nodes()).index(reference_id) + 1}</sub><br>'
    annotation_text += "Recommendations are ranked in order of preference, from bottom to top, with the bottom being the most preferred."

    # fig.add_annotation(
    #     text=annotation_text,
    #     align='left',
    #     showarrow=False,
    #     xref='paper',
    #     yref='paper',
    #     borderpad=6,
    #     bgcolor='black',
    #     bordercolor='mediumseagreen',
    #     borderwidth=1,
    #     font_size=13,
    #     font=dict(color='white', family='Arial, sans-serif', size=19),
    #     xanchor='left',
    #     yanchor='bottom'
    # )
    
    return fig
