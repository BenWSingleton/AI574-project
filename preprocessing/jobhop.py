import spacy
import pandas as pd
from datasets import load_dataset
from datetime import datetime
import numpy as np
import networkx as nx

nlp = spacy.load("en_core_web_sm")

def extract_skills(description):
    """Extract meaningful skills (action phrases and filtered nouns) from a job description."""
    if not isinstance(description, str) or description.lower() == "unknown":
        return []
    
    doc = nlp(description)
    skills = []
    
    # Extract action phrases
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ not in {"be", "have", "do"}:
            phrase = token.lemma_
            for child in token.children:
                if child.dep_ == "dobj":
                    phrase += " " + child.text
                    for subchild in child.children:
                        if subchild.dep_ in {"prep", "pobj"}:
                            phrase += " " + subchild.text
            if len(phrase.split()) > 1:
                skills.append(phrase)
    
    # Noun phrases
    for chunk in doc.noun_chunks:
        text = chunk.text.strip().lower()
        if (1 < len(text.split()) <= 4 and
            not text.startswith(("the ", "a ", "an ")) and
            text not in {"they", "it", "this", "all", "any"} and
            any(t.pos_ in {"NOUN", "PROPN"} for t in chunk)):
            skills.append(chunk.text)
    
    skills = sorted(list(set(skills)))
    return skills

# Load JobHop dataset (train split)
print("Loading JobHop dataset...")
ds = load_dataset("aida-ugent/JobHop", split="train")
df = ds.to_pandas()

# Handle null values
df['matched_code'] = df['matched_code'].fillna('unknown')

# Deduplicated view of unique occupations for skill extraction and ESCO mapping
unique_jobs = df[['matched_code', 'matched_label', 'matched_description']].drop_duplicates()
print(f"Found {len(unique_jobs)} unique matched_codes.")

# Derive skills from descriptions
print("Extracting skills with spaCy...")
unique_jobs['skills'] = unique_jobs['matched_description'].apply(extract_skills)
unique_jobs['skills_str'] = unique_jobs['skills'].apply(lambda x: ', '.join(x) if x else '')

# Integrate with ESCO official skills
esco_occupations = pd.read_csv('data/occupations_en.csv')
esco_skills = pd.read_csv('data/skills_en.csv')
esco_relations = pd.read_csv('data/occupationSkillRelations_en.csv')

# Create mapping of code -> list of skill labels
skill_map = {}
for code in unique_jobs['matched_code'].unique():
    if code == 'unknown':
        skill_map[code] = []
        continue
    occ_row = esco_occupations[esco_occupations['iscoGroup'].astype(str) == code]
    if not occ_row.empty:
        occ_uri = occ_row['conceptUri'].iloc[0]
        related_skills = esco_relations[esco_relations['occupationUri'] == occ_uri]
        skill_uris = related_skills['skillUri']
        skills_df = esco_skills[esco_skills['conceptUri'].isin(skill_uris)]
        essential = related_skills[related_skills['relationType'] == 'essential']['skillUri']
        optional = related_skills[related_skills['relationType'] == 'optional']['skillUri']
        skill_list = list(skills_df[skills_df['conceptUri'].isin(essential)]['preferredLabel']) + \
                     list(skills_df[skills_df['conceptUri'].isin(optional)]['preferredLabel'])
        skill_map[code] = sorted(set(skill_list))


unique_jobs['esco_skills'] = unique_jobs['matched_code'].apply(lambda x: skill_map.get(x, []))
unique_jobs['esco_skills_str'] = unique_jobs['esco_skills'].apply(lambda x: ', '.join(x))

# Save processed unique jobs with both NLP and ESCO skills
output_file = 'data/processed_unique_jobs_with_skills.csv'
unique_jobs[['matched_code', 'matched_label', 'matched_description', 'skills_str', 'esco_skills_str']].to_csv(output_file, index=False)
print(f"Saved processed data to '{output_file}'")

# Provide counts of occurrences per code
grouped_counts = df.groupby('matched_code').size().reset_index(name='count')
grouped_counts.to_csv('data/grouped_counts_by_code.csv', index=False)
print("Saved grouped counts to 'data/grouped_counts_by_code.csv'")

# Parse dates from Q# YYYY to month
def parse_quarter_date(qstr):
    if pd.isna(qstr) or qstr == 'null':
        return pd.NaT
    try:
        quarter, year = qstr.split()
        year = int(year)
        quarter_num = int(quarter[1])
        month = {1:1, 2:4, 3:7, 4:10}[quarter_num]
        return pd.Timestamp(f"{year}-{month:02d}-01")
    except:
        return pd.NaT

df['start_ts'] = df['start_date'].apply(parse_quarter_date)
df['end_ts'] = df['end_date'].apply(parse_quarter_date)
df['duration_months'] = ((df['end_ts'] - df['start_ts']).dt.days / 30).fillna(0)

# Sort and group by person_id
df_sorted = df.sort_values(['person_id', 'start_ts']).reset_index(drop=True)

# Create sequences of cumulative skills and role paths per person
def build_trajectory(group):
    group = group.sort_values('start_ts')
    cum_skills = []
    sequences = []
    for i, row in group.iterrows():
        skills = skill_map.get(row['matched_code'], [])
        cum_skills = list(set(cum_skills + skills)) if cum_skills else skills
        sequences.append({
            'person_id': row['person_id'],
            'role_sequence': ' -> '.join(group['matched_label'].tolist()[:i+1]),
            'cumulative_skills': cum_skills,
            'current_job_code': row['matched_code'],
            'next_job_code': group['matched_code'].iloc[i+1] if i+1 < len(group) else None,
            'total_experience_years': (group['end_ts'].max() - group['start_ts'].min()).days / 365 if len(group) > 1 else 0
        })
    return pd.DataFrame(sequences)

trajectories = df_sorted.groupby('person_id').apply(build_trajectory).reset_index(drop=True)
trajectories.to_csv('data/jobhop_trajectories.csv', index=False)
print(f"Generated {len(trajectories)} trajectories; avg length: {df_sorted.groupby('person_id').size().mean()}")

# Differentiate skill types for nuanced analysis
hard_keywords = {'python', 'sql', 'java', 'aws', 'certification'}  # Tech skills, can update as needed
def classify_skills(skills_list):
    hard = [s for s in skills_list if any(kw in s.lower() for kw in hard_keywords)]
    soft = [s for s in skills_list if s not in hard]
    return {'hard': hard, 'soft': soft}

trajectories['skills_classified'] = trajectories['cumulative_skills'].apply(classify_skills)