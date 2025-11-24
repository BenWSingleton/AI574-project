import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def found_score(true_missing, predicted_missing):
    scores = []
    for t, p in zip(true_missing, predicted_missing):
        t = set(t)
        p = set(p)
        score = len(t.intersection(p))/len(t) if len(t) > 0 else 0
        scores = scores + [score]
    return np.mean(scores)

def unnecessary_score(true_missing, predicted_missing):
    scores = []
    for t, p in zip(true_missing, predicted_missing):
        t = set(t)
        p = set(p)
        score = len(p - t)/len(p) if len(p) > 0 else 0
        scores = scores + [score]
    return np.mean(scores)

def redundant_score(predicted_missing, skills_present):
    scores = []
    for p, r in zip(predicted_missing, skills_present):
        p = set(p)
        r = set(r)
        
        score = len(p & r)/len(p) if len(p) > 0 else 0   
        scores = scores + [score]

    return np.mean(scores)

def get_metrics(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    overlap = set1 & set2
    difference = set1 ^ set2

    print(f"Overlap: {len(overlap)}")
    print(f"No Overlap: {len(difference)} ")

def presence_score(true_missing, predicted_missing):
    hits = 0
    total = len(true_missing)

    for t, p in zip(true_missing, predicted_missing):
        t = set(t)
        p = set(p)

        if len(t & p) > 0:
            hits += 1

    return hits / total if total > 0 else 0