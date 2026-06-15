import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    # Write code here
    N = len(docs)

    # Document lengths and average length
    doc_lens = [len(doc) for doc in docs]
    avgdl = np.mean(doc_lens)

    # Document frequency for each query term
    df = {}
    for term in set(query_tokens):
        df[term] = sum(term in doc for doc in docs)

    # BM25 IDF
    idf = {
        term: math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        for term, df_t in df.items()
    }

    scores = []

    for doc, dl in zip(docs, doc_lens):
        tf = Counter(doc)
        score = 0.0

        for term in query_tokens:
            if term not in tf:
                continue

            f = tf[term]

            denom = f + k1 * (1 - b + b * dl / avgdl)

            score += idf[term] * (
                f * (k1 + 1) / denom
            )

        scores.append(score)

    return np.array(scores, dtype=float)