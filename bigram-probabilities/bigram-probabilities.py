from collections import defaultdict

def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Your code here
    # Count bigrams
    counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for w in range(len(tokens) - 1):
        w1, w2 = tokens[w], tokens[w + 1]
        counts[(w1, w2)] += 1
        unigram_counts[w1] += 1

    # Vocabulary size
    vocab = set(tokens)
    V = len(vocab)

    # Compute smoothed probabilities
    probs = {}

    for w1 in vocab:
        for w2 in vocab:
            c_bigram = counts.get((w1, w2), 0)
            c_w1 = unigram_counts.get(w1, 0)

            probs[(w1, w2)] = (c_bigram + 1) / (c_w1 + V)

    return dict(counts), probs