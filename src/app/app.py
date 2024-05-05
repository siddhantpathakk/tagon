import plotly.figure_factory as ff
import streamlit as st
import numpy as np


@st.cache_data
def load_data(user_id):
    pass


def create_user_item_matrix(data):
    pass


def create_user_item_graph(data):
    pass


def predict(uid):
    pass


if __name__ == '__main__':
    st.set_page_config(page_title='TAGON', 
                       page_icon='random', 
                       layout='wide',
                       menu_items={"Get Help": "https://www.github.com/siddhantpathakk/tagon"}
                       )
    
    st.title('FYP Demonstration for TAGON: Temporal Attention Graph Neural Networks for Sequential Recommendation')
    
    user_id = st.text_input('User ID', '0')
    
    data_load_state = st.text('Loading data...')
    data = load_data(user_id)
    data_load_state.text("Done! (using st.cache_data)")
    
    if st.checkbox('Show user history (CSV format)'):
        st.subheader('CSV')
        st.write(data)
        
    st.subheader('Continuous Time Bipartite Graph CTBG for historical user-item interactions')
    user_item_matrix = create_user_item_matrix(data)
    ctbg_graph = create_user_item_graph(user_item_matrix)
    # st.plotly_chart(ctbg_graph)

    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ['Group 1', 'Group 2', 'Group 3']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])

    # Plot!
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    st.subheader('Predictions')
    predictions = None
    
    if st.button('Predict'):
        predictions = predict(user_id)
        st.write(predictions)
        
    if st.checkbox('Show output (CSV format)'):
        st.subheader('CSV')
        st.write(predictions)
