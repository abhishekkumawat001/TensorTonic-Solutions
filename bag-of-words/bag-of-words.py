import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    n = len(vocab)
    emb_vec = np.zeros(n, dtype=int)

    # Map each vocabulary word to its index
    wtoi = {word: i for i, word in enumerate(vocab)}

    # Count occurrences
    for token in tokens:
        if token in wtoi:
            emb_vec[wtoi[token]] +=1

    return emb_vec