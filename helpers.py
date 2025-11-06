import ast
import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_response(pompt, model="mistral:instruct"):
    response = ollama.generate(model=model, prompt=pompt)
    return response

def get_prompt(description, doc_type):
    if doc_type not in ['resume', 'job']:
        return "Invalid type"
    
    prompt = f"""Extract the skills from the following {doc_type}, return a comma-separated list. 
Do not include education and certifications.
Do not return the same skill more than one even if it's mentiond multiple times. 
Do not include years of experience or seniority levels, only the skill names.
Return only JSON array of strings.

    {description}"""

    return prompt

def get_skills(document, doc_type):
    prompt = get_prompt(document, doc_type)
    response = get_response(prompt)
    skills = ast.literal_eval(response['response'].strip())
    return skills

def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response['embedding']

def match_closed_skills(skills, esco_embeddings, top_k=5):
    skill_embeddings = [get_embedding(skill)['embedding'] for skill in skills]
    similarities = cosine_similarity(skill_embeddings, esco_embeddings)
    
    matched_skills = []
    for sim in similarities:
        top_indices = sim.argsort()[-top_k:][::-1]
        matched_skills.append(top_indices)
    
    return matched_skills

def process_row(row):
    prompt = get_prompt(row['Resume_str'], 'resume')
    response = get_response(prompt)
    skills = ast.literal_eval(response['response'].strip())
    return row.name, skills 

def get_list(data, type, max_workers=5):
    data = data.copy()
    data['extracted_skills'] = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor: 
        # Submit all tasks
        future_to_index = {executor.submit(process_row, row): row.name for index, row in data.iterrows()}
        
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

def match_closest_skills(skills, esco_skills, threhold=0.8, fill=False):
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

        if best_match_score >= threhold:
            scores.append(np.max(sim))
            matched_skills.append(esco_labels[best_match_idx])
        else:
            if fill:
                scores.append(None)
                matched_skills.append(None)

    return matched_skills, scores

def match_all_skills(data, esco_skills, threshold=0.8):
    data = data.copy()
    data['matched_skills'] = None

    for index, row in data.iterrows():
        skills = row['extracted_skills']
        matched = match_closest_skills(skills, esco_skills, threshold=0.8)
        data.at[index, 'matched_skills'] = matched

    return data