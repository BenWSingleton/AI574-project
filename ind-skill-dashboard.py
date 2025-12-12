import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
import json

def get_esco_skills(row):
    """
    Extract ESCO labels from 'matched_skills' column.
    Works with: list, np.ndarray, tuple, str, None, NaN.
    """
    if hasattr(row, '_fields'):
        raw = getattr(row, 'matched_skills', None)
    else:
        raw = row.get('matched_skills', None)

    if raw is None:
        return []

    if isinstance(raw, float) and pd.isna(raw):
        return []

    if isinstance(raw, (list, np.ndarray, tuple)):
        return [str(s).strip() for s in raw if str(s).strip()]

    if isinstance(raw, str):
        import ast
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except:
            pass
        return [s.strip() for s in raw.split(',') if s.strip()]

    return []

BASE = Path(".")
MATCH_DIR      = BASE / "matches"
CACHE_FILE     = MATCH_DIR / "skill_cache.json"
MATCH_DIR = Path("matches")
SUMMARY_FILE = MATCH_DIR / "ind_skills_scores_085.parquet"
resumes = pd.read_parquet('processed/resume_matched.parquet', columns=['ID', 'matched_skills'])
jobs = pd.read_parquet('processed/dice_job_descriptions_matched.parquet', columns=['uniq_id', 'jobtitle'])

skill_cache = {}
if CACHE_FILE.exists():
    skill_cache = {k: np.array(v) for k, v in json.load(open(CACHE_FILE)).items()}

def embed_skill(skill):
    skill = skill.strip()
    if skill in skill_cache:
        return skill_cache[skill]
    
    return None

def embed_list(skills):
    vectors = []
    for s in skills:
        vec = embed_skill(s)
        if vec is not None:
            vectors.append(vec)
    return np.vstack(vectors) if vectors else np.array([]).reshape(0, 768)

st.set_page_config(page_title="My Resume vs 22,000 Jobs", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet(SUMMARY_FILE)
    df['gaps'] = df['gaps'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    df['n_gaps'] = df['gaps'].str.len()
    df['composite_score'] = df['pct_job_covered'] * np.log1p(df['n_job_skills'])
    return df

df = load_data()

st.title("My Resume vs 22,000 Jobs: Gaps in Good Matches")

st.markdown("""
This dashboard focuses on **jobs you're close to qualifying for** (high coverage, but with gaps).  
It shows the top matches with gaps, and lets you explore how close you are to covering those gaps with your existing skills.
""")

# ── Resume selector ──
resume_ids = sorted(df["resume_id"].unique())
resume_id = st.selectbox("Select Your Resume ID", resume_ids, index=0)

if resume_id:
    matches = df[df["resume_id"] == resume_id].copy()
    
    st.header("Top Jobs Where You're Close (High Coverage, But Gaps)")
    st.markdown("These are jobs you're **70%+ qualified for** — sorted by composite score (coverage + complexity). We skip perfect matches.")

    top_n = st.slider("Show top N jobs with gaps", 5, 30, 15)

    high_coverage_with_gaps = matches[(matches['pct_job_covered'] >= 0.7) & (matches['n_gaps'] > 0)]
    top_jobs = high_coverage_with_gaps.sort_values('composite_score', ascending=False).head(top_n)

    jobs_df = jobs.rename(columns={'uniq_id': 'job_id'})

    top_jobs_with_titles = top_jobs.merge(
        jobs_df,
        on='job_id',
        how='left'
    )

    st.success(f"Found {len(high_coverage_with_gaps)} jobs with high score, but not perfect. Showing {len(top_jobs)} jobs.")

    if len(top_jobs) == 0:
        st.info("No high-coverage jobs with gaps found — your resume is a perfect fit for everything!")
    else:
        st.dataframe(
            top_jobs_with_titles[['jobtitle', 'pct_job_covered', 'n_job_skills', 'n_gaps', 'composite_score']].round(3),
            use_container_width=True,
            hide_index=True
        )

    st.header("Missing Skills Blocking These Jobs")

    all_gaps = [gap for gaps in top_jobs['gaps'] for gap in gaps]
    gap_freq = pd.Series(all_gaps).value_counts().reset_index()
    gap_freq.columns = ['Missing Skill', 'Number of Jobs']

    fig_bar = px.bar(
        gap_freq,
        x='Number of Jobs',
        y='Missing Skill',
        orientation='h',
        title="Top Missing Skills in Your Best Matches",
        height=max(600, len(gap_freq) * 28),
        text='Number of Jobs',
        color='Number of Jobs',
        color_continuous_scale='Reds'
    )

    fig_bar.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=300, r=50, t=80, b=50),
        yaxis_tickmode='linear',
        uniformtext=dict(mode='hide'),
        showlegend=False
    )

    fig_bar.update_yaxes(
        tickmode='array',
        tickvals=gap_freq['Missing Skill'],
        ticktext=gap_freq['Missing Skill'],
        automargin=True,
        title=None
    )

    fig_bar.update_yaxes(
        tickfont=dict(size=11),
        ticklabeloverflow="allow"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.header("How Close Are Your Skills to These Gaps?")
    st.markdown("For each missing skill, see the **highest similarity** to any of your existing skills.")

    resume_row = resumes[resumes['ID'] == resume_id].iloc[0]
    your_skills = get_esco_skills(resume_row)
    your_emb = embed_list(your_skills)

    unique_gaps = list(set(all_gaps))
    if unique_gaps:
        gap_emb = embed_list(unique_gaps)
        sim = cosine_similarity(your_emb, gap_emb)

        max_sim_per_gap = sim.max(axis=0)
        gap_sim = pd.DataFrame({
            'Missing Skill': unique_gaps,
            'Your Closest Skill': [your_skills[sim[:, j].argmax()] for j in range(len(unique_gaps))],
            'Similarity': max_sim_per_gap
        }).sort_values('Similarity', ascending=False).reset_index(drop=True)

        st.dataframe(gap_sim.round(3), use_container_width=True, hide_index=True)
    else:
        st.info("No gaps in your top matches — you're fully qualified!")

else:
    st.info("Select a resume ID to begin.")