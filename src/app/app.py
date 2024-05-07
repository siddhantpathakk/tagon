import os
import streamlit as st
import pandas as pd

from plot import make_ctbg
from src.components.trainer.trainer import Trainer
from src.components.utils.parse import parse_training_args
from src.components.trainer.trainer_utils import setup_model, setup_optimizer
from src.components.utils.utils import EarlyStopMonitor, set_seed
from src.components.data.data import Data
from src.components.utils.consts import *

@st.cache_data
def get_output(dataset, user_id):
    args = parse_training_args(dataset=dataset)

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    USE_TIME = args.time
    AGG_METHOD = args.agg_method
    ATTN_MODE = args.attn_mode
    SEQ_LEN = NUM_NEIGHBORS
    DATASET = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    infer_mode = True 
    USER_ID = user_id

    cwd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    set_seed(args.seed)

    data = Data(DATASET, args)
    n_nodes = data.max_idx
    n_edges = data.num_total_edges
    print('Number of nodes:', n_nodes)
    print('Number of edges:', n_edges)

    user_history = data.get_user_history(USER_ID)
    print('User history', '\n', user_history)

    pretrain = pretrain_app_path_slab(args, cwd)

    model = setup_model(data, args, data.max_idx, GPU, NUM_LAYER, USE_TIME, AGG_METHOD, ATTN_MODE, SEQ_LEN, NUM_HEADS, DROP_OUT, NODE_DIM, TIME_DIM,
                        load_pretrain=pretrain)
    optimizer = setup_optimizer(model, LEARNING_RATE, load_pretrain=pretrain)
    early_stopper = EarlyStopMonitor(max_round=5, higher_better=True)
    trainer = Trainer(data, model, optimizer, early_stopper,
                      NUM_EPOCH, BATCH_SIZE, args)

    if infer_mode:
        result, output = trainer.test(user_id=USER_ID)
        print(result)
        print(output)   
    
    return user_history, output



# Streamlit UI
st.set_page_config(page_title='TAGON',
                   page_icon=":material/sdk:",
                    layout='centered',
                    initial_sidebar_state='expanded',
                    menu_items={
                        "Get Help": "https://www.github.com/siddhantpathakk/tagon",
                        'About': "This is a final year project demonstration for TAGON."
                        }
                    )

st.title(f'FYP Demonstration for TAGON')

dataset_names = ['ml-100k', 'Baby', 'Digital_Music', 'Toys_and_Games', 'Tools_and_Home_Improvement']

# Sidebar
st.sidebar.title("Settings")

st.sidebar.subheader("Select dataset and user ID")
selected_dataset = st.sidebar.selectbox("Select dataset", dataset_names, placeholder="Select a dataset")
user_id = st.sidebar.text_input("Enter User ID", "2")
user_id = int(user_id)


st.sidebar.subheader("CTBG Settings")
trim = st.sidebar.slider("Trim (for CTBG)", 1, 6, 4)
show_csv = st.sidebar.checkbox("Show CSV Data")

session_state = False
st.sidebar.subheader("Make Predictions")
if st.sidebar.button('Predict', type='primary'):
    print(f"Selected dataset: {selected_dataset}")
    print(f"User ID: {user_id}")
    user_hist, output = get_output(selected_dataset, user_id)
    session_state = True
    st.toast("Predictions completed!",  icon="✅")
    
if session_state:
    
    st.header("Continuous Time Bipartite Graph (CTBG)")
    fig = make_ctbg(user_hist, selected_dataset, trim=trim)
    st.plotly_chart(fig)
        
        
    if show_csv:
        dataframe = pd.DataFrame(user_hist)
        st.dataframe(dataframe)

    st.header("Recommendations")
    if show_csv:
        st.subheader("CSV Data")
        dataframe = pd.DataFrame(output)
        dataframe = dataframe.sort_values(by='timestamp', ascending=False)
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s')
        dataframe = dataframe[['u_pos_gd', 'timestamp', 'predicted']][::-1].reset_index(drop=True)
        st.dataframe(dataframe)