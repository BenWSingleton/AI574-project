import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from streamlit_plotly_events import plotly_events

st.title("Resume-Job Embedding Visualization")
st.set_page_config(page_title="Resume-Job Visualization", layout="wide")

df = pd.read_parquet('processed/dim_reduced.parquet')

df = df.sample(n=5, random_state=44).reset_index(drop=True)

columns = ['matched_skills_ordered', 'best_match_job_skills']

df

skill_xyz   = np.vstack(df["skill_tsne"].values)
job_xyz     = np.vstack(df["job_tsne"].values)
diff_xyz    = np.vstack(df["diff_tsne"].values)
predict_xyz = np.vstack(df["predict_tsne"].values)

skill_text   = df["matched_skills_ordered"].apply(lambda x: "<br>".join(x) if isinstance(x, list) else str(x)).values
job_text     = df["best_match_job_skills"].apply(lambda x: "<br>".join(x) if isinstance(x, list) else str(x)).values
predict_text = df["predicted_missing"].apply(lambda x: "<br>".join(x) if isinstance(x, list) else str(x)).values

fig = go.Figure()

# Scatter points for each embedding type
fig.add_trace(go.Scatter3d(
    x=skill_xyz[:,0], y=skill_xyz[:,1], z=skill_xyz[:,2],
    mode="markers",
    marker=dict(size=6, color="blue"),
    name="Skill TSNE",
    text=skill_text,
    hovertemplate="%{text}<extra></extra>"
))

fig.add_trace(go.Scatter3d(
    x=job_xyz[:,0], y=job_xyz[:,1], z=job_xyz[:,2],
    mode="markers",
    marker=dict(size=6, color="green"),
    name="Job TSNE",
    text=job_text,
    hovertemplate="%{text}<extra></extra>"
))

fig.add_trace(go.Scatter3d(
    x=diff_xyz[:,0], y=diff_xyz[:,1], z=diff_xyz[:,2],
    mode="markers",
    marker=dict(size=6, color="orange"),
    name="Diff TSNE"
))

fig.add_trace(go.Scatter3d(
    x=predict_xyz[:,0], y=predict_xyz[:,1], z=predict_xyz[:,2],
    mode="markers",
    marker=dict(size=6, color="red"),
    name="Predicted Missing TSNE",
    text=predict_text,
    hovertemplate="%{text}<extra></extra>"
))

# ---- CONNECTING LINES ----
for i in range(len(df)):
    # skill → diff
    fig.add_trace(go.Scatter3d(
        x=[skill_xyz[i,0], diff_xyz[i,0]],
        y=[skill_xyz[i,1], diff_xyz[i,1]],
        z=[skill_xyz[i,2], diff_xyz[i,2]],
        mode="lines",
        line=dict(color="gray", width=2),
        showlegend=False
    ))

    # diff → job
    fig.add_trace(go.Scatter3d(
        x=[diff_xyz[i,0], job_xyz[i,0]],
        y=[diff_xyz[i,1], job_xyz[i,1]],
        z=[diff_xyz[i,2], job_xyz[i,2]],
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))

    # diff → predict
    fig.add_trace(go.Scatter3d(
        x=[diff_xyz[i,0], predict_xyz[i,0]],
        y=[diff_xyz[i,1], predict_xyz[i,1]],
        z=[diff_xyz[i,2], predict_xyz[i,2]],
        mode="lines",
        line=dict(color="red", width=2, dash="dot"),
        showlegend=False
    ))

# Layout
fig.update_layout(
    width=1200,
    height=800,
    scene=dict(
        xaxis_title="TSNE X",
        yaxis_title="TSNE Y",
        zaxis_title="TSNE Z",
    ),
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)
