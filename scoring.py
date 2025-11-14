import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def intersection_score(skills1, skills2):
    if len(skills1) != len(skills2):
        return "Can't match "
    
    total_score = 0
    count = 0

    for group1, group2 in zip(skills1, skills2):
        set1, set2 = set(group1), set(group2)
        overlap = len(set1 & set2)
        union = len(set1 | set2)
        score = overlap / union if union else 0
        total_score += score
        count += 1

    return (total_score / count) if count > 0 else 0.0