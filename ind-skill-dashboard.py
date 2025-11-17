import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Individual ESCO Skill Matching Dashboard", layout="wide")

MATCH_DIR = Path("matches")
SUMMARY_FILE = MATCH_DIR / "ind_skills_scores.parquet"

st.title("Resume ↔ Job Matching Dashboard")
st.sidebar.header("Controls")

# Load data
@st.cache_data
def load_data():
    df = pd.read_parquet(SUMMARY_FILE)
    return df

df = load_data()

st.sidebar.metric("Total Resume–Job Pairs Processed", len(df))
st.sidebar.metric("Resumes in dataset", df["resume_id"].nunique())
st.sidebar.metric("Jobs in dataset", df["job_id"].nunique())

# ── Top N matches per resume ──
top_n = st.sidebar.slider("Show top N jobs per resume", 3, 20, 10)

best_matches = (
    df.sort_values(["resume_id", "pct_covered", "avg_match"], ascending=[True, False, False])
      .groupby("resume_id")
      .head(top_n)
      .reset_index(drop=True)
)

st.header(f"Top {top_n} Job Matches per Resume")
st.dataframe(
    best_matches[["resume_id", "job_id", "pct_covered", "avg_match", "n_matches"]].round(4),
    use_container_width=True,
    hide_index=True
)

# ── Skill Coverage Distribution ──
st.header("Skill Coverage Distribution")
fig_cov = px.histogram(df, x="pct_covered", nbins=50, title="How much of the resume skills are covered by the job?")
fig_cov.update_layout(xaxis_title="Pct Covered (resume skills → job)", yaxis_title="Count")
st.plotly_chart(fig_cov, use_container_width=True)

# ── Average Similarity Distribution ──
fig_sim = px.histogram(df, x="avg_match", nbins=50, title="Average cosine similarity of matched skills")
st.plotly_chart(fig_sim, use_container_width=True)

# ── Search by Resume ID ──
st.header("Search Specific Resume")
resume_id = st.text_input("Enter Resume ID")
if resume_id:
    matches = df[df["resume_id"] == resume_id].sort_values("pct_covered", ascending=False)
    if len(matches) == 0:
        st.warning("No matches found for this resume ID")
    else:
        st.success(f"Found {len(matches)} job matches")
        st.dataframe(matches[["job_id", "pct_covered", "avg_match", "n_matches", "gaps"]].head(20))

# ── Skill Gap Explorer ──
st.header("Most Common Skill Gaps (across all pairs)")
all_gaps = pd.Series([gap for sublist in df["gaps"] for gap in sublist])
top_gaps = all_gaps.value_counts().head(30)

fig_gaps = px.bar(
    x=top_gaps.values,
    y=top_gaps.index,
    orientation='h',
    title="Top 30 Most Frequently Missing ESCO Skills",
    labels={"x": "Times missing", "y": "ESCO Skill"}
)
fig_gaps.update_layout(height=800)
st.plotly_chart(fig_gaps, use_container_width=True)

# ── Download results ──
st.download_button(
    label="Download Full Results as CSV",
    data=df.to_csv(index=False).encode(),
    file_name="esco_matching_results_full.csv",
    mime="text/csv"
)
