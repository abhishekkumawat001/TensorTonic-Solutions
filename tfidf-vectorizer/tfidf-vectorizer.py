import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    tokenized_docs = [doc.lower().split() for doc in documents]

    # build sorted vocab
    vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))
    vocab_idx = {word: i for i , word in enumerate(vocabulary)}

    N = len(documents)
    V = len(vocabulary)

    # document freq 
    
    df = Counter()

    for doc in tokenized_docs:
        for word in set(doc):
            df[word] += 1

    # inverse doc freq 
    idf = {
        word: math.log(N/df[word])
        for word in vocabulary
    }
    #tfidf matrix
    tfidf_matrix = np.zeros((N,V))

    for doc_idx, doc in enumerate(tokenized_docs):
        word_counts = Counter(doc)
        doc_len = len(doc)

        for word, count in word_counts.items():
            tf = count / doc_len
            tfidf_matrix[doc_idx, vocab_idx[word]] = tf * idf[word]

    return tfidf_matrix, vocabulary
    
            
    