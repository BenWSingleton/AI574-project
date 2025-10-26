import pandas as pd
import re
import spacy

# Better to load dataset from csv because HuggingFace has a few bad columns
# Download ResumeTrain.csv from https://huggingface.co/datasets/Divyaamith/resume-dataset/resolve/main/ResumeTrain.csv
df = pd.read_csv('ResumeTrain.csv', on_bad_lines='skip', usecols=range(4), names=['ID', 'Resume_str', 'Resume_html', 'Category'], header=None, low_memory=False)

print(f"Dataset shape: {df.shape}")
print(df.head())

# Clean the text in Resume_str column
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

df['cleaned_resume_str'] = df['Resume_str'].apply(clean_text)

# Extract skills using spaCy
nlp = spacy.load('en_core_web_sm')

def extract_skills_spacy(text):
    doc = nlp(text)
    skills = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    return list(set(skills))

df['extracted_skills_spacy'] = df['cleaned_resume_str'].apply(extract_skills_spacy)

# Match to ESCO skills
esco_df = pd.read_csv('esco_skills.csv')
esco_skills = set(esco_df['preferredLabel'].str.lower().unique())

def match_esco_skills(extracted_skills):
    matched = [skill for skill in extracted_skills if skill.lower() in esco_skills]
    return matched

df['esco_matched_skills'] = df['extracted_skills_spacy'].apply(match_esco_skills)

# Save preprocessed dataset
df.to_csv('preprocessed_resumes.csv', index=False)

print("Preprocessing complete. Saved to 'preprocessed_resumes.csv'")