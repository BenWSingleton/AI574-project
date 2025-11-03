import ast

import ollama

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
    return response