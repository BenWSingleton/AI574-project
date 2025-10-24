import pandas as pd
import spacy
from datasets import load_dataset

nlp = spacy.load("en_core_web_sm")

def extract_skills(description):
    """Extract potential skills (verbs and noun phrases) from a job description."""
    if not isinstance(description, str) or description.lower() == "unknown":
        return []
    
    doc = nlp(description)
    skills = []
    
    # Extract verbs
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ not in {"be", "have", "do"}:
            skills.append(token.lemma_)
    
    # Extract noun phrases
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3 and chunk.text.lower() not in {"they", "it", "this"}:
            skills.append(chunk.text)
    
    skills = sorted(list(set(skills)))
    return skills

# Load JobHop dataset (train split)
print("Loading JobHop dataset...")
ds = load_dataset("aida-ugent/JobHop", split="train")
df = ds.to_pandas()

# Clean nulls in matched_code
df['matched_code'] = df['matched_code'].fillna('unknown')

# Group by matched_code and get unique jobs with descriptions
unique_jobs = df[['matched_code', 'matched_label', 'matched_description']].drop_duplicates()
print(f"Found {len(unique_jobs)} unique matched_codes.")

# Extract skills from matched_description
print("Extracting skills with spaCy...")
unique_jobs['skills'] = unique_jobs['matched_description'].apply(extract_skills)
unique_jobs['skills_str'] = unique_jobs['skills'].apply(lambda x: ', '.join(x) if x else '')

# Save processed data
output_file = 'processed_unique_jobs_with_nlp_skills.csv'
unique_jobs[['matched_code', 'matched_label', 'matched_description', 'skills_str']].to_csv(output_file, index=False)
print(f"Saved processed data to '{output_file}'")

# Grouped stats
grouped_counts = df.groupby('matched_code').size().reset_index(name='count')
grouped_counts.to_csv('grouped_counts_by_code.csv', index=False)
print("Saved grouped counts to 'grouped_counts_by_code.csv'")