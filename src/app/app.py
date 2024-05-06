import os
import streamlit as st
import pandas as pd
import plotly.express as px

from src.components.trainer.trainer import Trainer
from src.components.utils.parse import parse_app_args
from src.components.trainer.trainer_utils import setup_model
from src.components.utils.utils import set_seed
from src.components.data.data import Data


def get_user_hist(dataset, user_id):
    configfile_path = f"src/config/{dataset}.json"
    args = parse_app_args(configfile_path)
    DATASET = args.data
    USER_ID = user_id
    set_seed(args.seed)
    data = Data(DATASET, args)
    user_history = data.get_user_history(USER_ID)
    return user_history


def get_output(dataset, user_id):
    configfile_path = f"src/config/{dataset}.json"
    args = parse_app_args(configfile_path)

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
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    USER_ID = user_id

    set_seed(args.seed)

    data = Data(DATASET, args)

    pretrain = f'src/components/tmp/ckpts/slab/{DATASET}/{DATASET}_TARGON.pt'

    model = setup_model(data, args, data.max_idx, GPU, NUM_LAYER, USE_TIME, AGG_METHOD,
                        ATTN_MODE, SEQ_LEN, NUM_HEADS, DROP_OUT, NODE_DIM, TIME_DIM,
                        load_pretrain=pretrain)

    trainer = Trainer(data, model,
                      optimizer=None, early_stopper=None,
                      NUM_EPOCH=NUM_EPOCH, BATCH_SIZE=BATCH_SIZE, args=args)

    _, output = trainer.test(user_id=USER_ID)
    return output

# Streamlit UI
st.set_page_config(page_title='TAGON',
                    page_icon='random',
                    layout='wide',
                    menu_items={
                        "Get Help": "https://www.github.com/siddhantpathakk/tagon",
                        'About': "This is a final year project demonstration for TAGON."
                        }
                    )

st.title(f'FYP Demonstration for TAGON')

dataset_names = ['ml-100k', 'Baby', 'Digital_Music', 'Toys_and_Games', 'Tools_and_Home_Improvement']

selected_dataset = st.sidebar.selectbox("Select dataset", dataset_names, placeholder="Select a dataset")
if st.sidebar.button('Load Dataset'):
    print(f'Selected dataset: {selected_dataset}')
    st.toast("Dataset loaded!",  icon="✅")

user_id = st.sidebar.text_input("Enter User ID", "0")
if st.sidebar.button('Load user data'):
    user_id = int(user_id) 
    print(f'User ID: {user_id}')
    st.toast("User data loaded!",  icon="✅")

if st.sidebar.button('Predict'):
    st.toast("Predictions completed!",  icon="✅")

st.header("Continuous Time Bipartite Graph (CTBG)")
# # fig = px.line(load_dataset('example'), x='time', y='value')
# # st.plotly_chart(fig)

if st.checkbox('View in CSV format', key='user_data'):
    st.header("CSV Data")
    dataframe = pd.DataFrame(get_user_hist(selected_dataset, user_id))
    st.dataframe(dataframe)

st.header("Recommendations")
# # recs_figure = px.bar(load_user_data('example_user'), x='event', y='timestamp')
# # st.plotly_chart(recs_figure)

if st.checkbox('View in CSV format', key='recs'):
    st.header("CSV Data")
    dataframe = pd.DataFrame(get_output(selected_dataset, user_id))
    st.dataframe(dataframe)