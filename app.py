import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("Resume-Job Embedding Visualization")
st.set_page_config(page_title="Resume-Job Visualization", layout="wide")

if "df" not in st.session_state:
    np.random.seed(42)
    data = np.random.randn(5, 3)
    labels = np.random.choice(["A", "B", "C"], size=5)
    st.session_state.df = pd.DataFrame(data, columns=["x", "y", "z"])
    st.session_state.df["label"] = labels

user_text = st.text_input("Enter a label or description")
add_button = st.button("Add point")

if add_button and user_text:
    # For now, generate a random 3D location
    new_point = pd.DataFrame(
        [[np.random.randn(), np.random.randn(), np.random.randn(), user_text]],
        columns=["x", "y", "z", "label"]
    )
    st.session_state.df = pd.concat([st.session_state.df, new_point], ignore_index=True)

fig = px.scatter_3d(
    st.session_state.df, x="x", y="y", z="z",
    color="label",
    opacity=0.8,
    symbol="label",
    title="Embeddings in 3D"
)
fig.update_layout(height=800, width=1200) # Adjust the height and width as needed
fig.update_traces(marker=dict(size=10)) # Adjust the size of the markers
st.plotly_chart(fig, use_container_width=True)