import ast
import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

def get_response(prompt, model="mistral:instruct"):
    response = ollama.generate(model=model, prompt=prompt)
    return response

def get_prompt(description, doc_type):
    if doc_type not in ['resume', 'job']:
        return "Invalid type"
    
    prompt = f"""
You are an expert at identifying skills in text.

Your task:
Extract only professional skills (e.g., programming languages, tools, technical abilities, and soft skills) from the following {doc_type}.

Rules:
- Return a valid Python list of strings, like ["Python", "Project Management", "Data Analysis"].
- Each skill should be a concise name (1–4 words).
- Do NOT include:
  - Education degrees (e.g., "Bachelor's in Computer Science")
  - Certifications (e.g., "AWS Certified")
  - Job titles or companies
  - Years of experience, seniority, or proficiency levels
- Do not include duplicates (case-insensitive).
- Return only the list, with no explanation or extra text.

Text:
\"\"\"{description}\"\"\"
"""
    return prompt


def get_skills(document, doc_type, model="mistral:instruct"):
    prompt = get_prompt(document, doc_type, model=model)
    response = get_response(prompt)
    skills = ast.literal_eval(response['response'].strip())
    return skills

def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response['embedding']

def get_batch_embeddings(input):
    response = ollama.embed(model="nomic-embed-text", input=input)
    return response

def embed_skills_list(matched_skills):
    joined_matched_skills = [','.join(skills) for skills in matched_skills] 
    response = get_batch_embeddings(joined_matched_skills)
    return response

def match_closed_skills(skills, esco_embeddings, top_k=5):
    skill_embeddings = [get_embedding(skill)['embedding'] for skill in skills]
    similarities = cosine_similarity(skill_embeddings, esco_embeddings)
    
    matched_skills = []
    for sim in similarities:
        top_indices = sim.argsort()[-top_k:][::-1]
        matched_skills.append(top_indices)
    
    return matched_skills

def process_row(row, col, doc_type, model="mistral:instruct"):
    prompt = get_prompt(row[col], doc_type)
    response = get_response(prompt, model)
    skills = ast.literal_eval(response['response'].strip())
    return row.name, skills 

def get_list(data, col, doc_type, max_workers=5, model="mistral:instruct"):
    data = data.copy()
    data['extracted_skills'] = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor: 
        # Submit all tasks
        future_to_index = {executor.submit(process_row, row, col, doc_type, model): row.name for index, row in data.iterrows()}
        
        # Process completed tasks with progress bar
        for future in tqdm.tqdm(as_completed(future_to_index), total=len(data)):
            try:
                index, skills = future.result(timeout=120)
                data.at[index, 'extracted_skills'] = skills
            except Exception as e:
                index = future_to_index[future]
                print(f"Error on row {index}: {e}")
                data.at[index, 'extracted_skills'] = []
    return data

def fill_missing_skills(data, skills_col, doc_type):
    data = data.copy()

    for index, row in data.iterrows():
        if len(row['extracted_skills']) == 0:
            try:
                print(f"Filling skills for row {index}")
                prompt = get_prompt(row[skills_col], doc_type)
                response = get_response(prompt)
                skills = ast.literal_eval(response['response'].strip())
                data.at[index, 'extracted_skills'] = skills
            except Exception as e:
                print(f"Error on row {index}: {e}")
                data.at[index, 'extracted_skills'] = []
    return data

def match_closest_skills(skills, esco_skills, threshold=0.8, fill=False):
    """
    Given a list of skill names and a dataframe of ESCO skills (with columns
    'preferredLabel' and 'embeddings'), return a list of the closest ESCO skill for each.
    """
    matched_skills = []
    scores = []

    # Extract ESCO data
    esco_embeddings = np.vstack(esco_skills['embeddings'].to_numpy())
    esco_labels = esco_skills['preferredLabel'].tolist()

    # Assume you have a function get_embedding(skill) that returns the same kind of embedding
    for skill in skills:
        emb = get_embedding(skill)  # You’ll need to define this part
        sim = cosine_similarity([emb], esco_embeddings)[0]
        best_match_score = np.max(sim)
        best_match_idx = np.argmax(sim)

        if best_match_score >= threshold:
            scores.append(np.max(sim))
            matched_skills.append(esco_labels[best_match_idx])
        else:
            if fill:
                scores.append(None)
                matched_skills.append(None)

    return matched_skills, scores

def process_row_skills(args):
    """Helper for multiprocessing — matches skills for one row."""
    index, row, esco_skills, threshold = args
    matched, _ = match_closest_skills(row['extracted_skills'], esco_skills, threshold=threshold)
    return index, matched

def match_all_skills(data, esco_skills, threshold=0.8):
    data = data.copy()
    data['matched_skills'] = None

    for index, row in tqdm.tqdm(data.iterrows(), total=len(data)):
        skills = row['extracted_skills']
        matched, _ = match_closest_skills(skills, esco_skills, threshold=threshold)
        data.at[index, 'matched_skills'] = matched

    return data

def match_all_skills_con(data, esco_skills, threshold=0.8, max_workers=4):
    data = data.copy()
    data['matched_skills'] = None

    tasks = [
        (index, row, esco_skills, threshold)
        for index, row in data.iterrows()
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row_skills, t) for t in tasks]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            index, matched = future.result()
            data.at[index, 'matched_skills'] = matched

    return data