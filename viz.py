import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from streamlit_plotly_events import plotly_events

st.title("Resume-Job Embedding Visualization")
st.set_page_config(page_title="Resume-Job Visualization", layout="wide")

df = pd.read_parquet('processed/dim_reduced.parquet')

df = df.sample(n=5, random_state=41).reset_index(drop=True)

columns = ['matched_skills_ordered', 'best_match_job_skills']

df

skill_xyz   = np.vstack(df["skill_tsne"].values)
job_xyz     = np.vstack(df["job_tsne"].values)
predict_xyz = np.vstack(df["predict_tsne"].values)

skill_text   = df["matched_skills_ordered"].apply(lambda x: "<br>".join(x) if isinstance(x, list) else str(x)).values
job_text     = df["best_match_job_skills"].apply(lambda x: "<br>".join(x) if isinstance(x, list) else str(x)).values
predict_text = df["predicted_missing"].apply(lambda x: "<br>".join(x) if isinstance(x, list) else str(x)).values

fig = go.Figure()

# --- Add scatter points ---
fig.add_trace(go.Scatter3d(
    x=skill_xyz[:, 0], y=skill_xyz[:, 1], z=skill_xyz[:, 2],
    mode="markers",
    marker=dict(size=4, opacity=0.8),
    name="Resume Skill Embeddings",
    customdata=np.array(skill_text).reshape(-1, 1),
    hovertemplate="<b>Resume Skills:</b><br>%{customdata[0]}<extra></extra>"
))

# --- Job Embeddings ---
fig.add_trace(go.Scatter3d(
    x=job_xyz[:, 0], y=job_xyz[:, 1], z=job_xyz[:, 2],
    mode="markers",
    marker=dict(size=4, opacity=0.8),
    name="Closest Job Embeddings",
    customdata=np.array(job_text).reshape(-1, 1),
    hovertemplate="<b>Job Skills:</b><br>%{customdata[0]}<extra></extra>"
))

# --- Predicted Missing Embeddings ---
fig.add_trace(go.Scatter3d(
    x=predict_xyz[:, 0], y=predict_xyz[:, 1], z=predict_xyz[:, 2],
    mode="markers",
    marker=dict(size=4, opacity=0.8),
    name="Predicted Missing Skill Embeddings",
    customdata=np.array(predict_text).reshape(-1, 1),
    hovertemplate="<b>Predicted Missing Skills:</b><br>%{customdata[0]}<extra></extra>"
))

# --- Lines connecting skill → job → predict ---
for s, j, p in zip(skill_xyz, job_xyz, predict_xyz):
    fig.add_trace(go.Scatter3d(
        x=[s[0], j[0], p[0]],
        y=[s[1], j[1], p[1]],
        z=[s[2], j[2], p[2]],
        mode="lines",
        line=dict(width=2),
        showlegend=False,
        hoverinfo="skip"    # do not show hover on lines
    ))

fig.update_layout(
    width=1100,
    height=800,
    scene=dict(
        xaxis_title='TSNE-1',
        yaxis_title='TSNE-2',
        zaxis_title='TSNE-3'
    ),
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig, use_container_width=True)