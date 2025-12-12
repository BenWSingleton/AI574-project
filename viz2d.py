import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Resume-Job Visualization (2D)", layout="wide")
st.title("Resume–Job Embedding Visualization (2D)")

df = pd.read_parquet('processed/dim_reduced.parquet')

df = df.sample(n=5, random_state=44).reset_index(drop=True)

# Display source data
df

# --- Extract 2D embeddings ---
skill_xy   = np.vstack(df["skill_tsne"].values)[:, :2]
job_xy     = np.vstack(df["job_tsne"].values)[:, :2]
diff_xy    = np.vstack(df["diff_tsne"].values)[:, :2]
predict_xy = np.vstack(df["predict_tsne"].values)[:, :2]

# --- Hover text ---
skill_text   = df["matched_skills_ordered"].apply(lambda x: "<br>".join(x) if isinstance(x, list) else str(x)).values
job_text     = df["best_match_job_skills"].apply(lambda x: "<br>".join(x) if isinstance(x, list) else str(x)).values
predict_text = df["predicted_missing"].apply(lambda x: "<br>".join(x) if isinstance(x, list) else str(x)).values

fig = go.Figure()

# ---- POINTS ----
fig.add_trace(go.Scatter(
    x=skill_xy[:,0], y=skill_xy[:,1],
    mode="markers",
    marker=dict(size=8, color="blue"),
    name="Skill TSNE",
    text=skill_text,
    hovertemplate="%{text}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=job_xy[:,0], y=job_xy[:,1],
    mode="markers",
    marker=dict(size=8, color="green"),
    name="Job TSNE",
    text=job_text,
    hovertemplate="%{text}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=diff_xy[:,0], y=diff_xy[:,1],
    mode="markers",
    marker=dict(size=8, color="orange"),
    name="Diff TSNE"
))

fig.add_trace(go.Scatter(
    x=predict_xy[:,0], y=predict_xy[:,1],
    mode="markers",
    marker=dict(size=8, color="red"),
    name="Predicted Missing TSNE",
    text=predict_text,
    hovertemplate="%{text}<extra></extra>"
))

# ---- CONNECTING LINES ----
for i in range(len(df)):
    sx, sy = skill_xy[i]
    jx, jy = job_xy[i]
    dx, dy = diff_xy[i]
    px, py = predict_xy[i]

    # skill → diff
    fig.add_trace(go.Scatter(
        x=[sx, dx], y=[sy, dy],
        mode="lines",
        line=dict(color="gray", width=2),
        showlegend=False
    ))

    # diff → job
    fig.add_trace(go.Scatter(
        x=[dx, jx], y=[dy, jy],
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False
    ))

    # diff → predict
    fig.add_trace(go.Scatter(
        x=[dx, px], y=[dy, py],
        mode="lines",
        line=dict(color="red", width=2, dash="dot"),
        showlegend=False
    ))

# ---- Layout ----
fig.update_layout(
    width=1200,
    height=800,
    xaxis_title="TSNE X",
    yaxis_title="TSNE Y",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)
