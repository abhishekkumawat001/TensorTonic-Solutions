from collections import Counter
import math

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.

    Args:
        candidate: list of tokens (candidate translation)
        reference: list of tokens (reference translation)
        max_n: maximum n-gram order

    Returns:
        float: BLEU score
    """

    def get_ngrams(tokens, n):
        """Return Counter of n-grams."""
        return Counter(
            tuple(tokens[i:i+n])
            for i in range(len(tokens) - n + 1)
        )

    c = len(candidate)  # candidate length
    r = len(reference)  # reference length

    precisions = []

    # Compute modified precision for each n-gram order
    for n in range(1, max_n + 1):
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)

        total_count = sum(cand_ngrams.values())

        # If candidate is shorter than n
        if total_count == 0:
            return 0.0

        clipped_count = 0
        for ng in cand_ngrams:
            clipped_count += min(
                cand_ngrams[ng],
                ref_ngrams.get(ng, 0)
            )

        p_n = clipped_count / total_count

        # If any precision is zero → BLEU = 0
        if p_n == 0:
            return 0.0

        precisions.append(p_n)

    # Brevity Penalty (BP)
    if c >= r:
        bp = 1.0
    else:
        bp = math.exp(1 - r / c)

    # Geometric mean of precisions
    log_sum = sum(math.log(p) for p in precisions)
    geo_mean = math.exp(log_sum / max_n)

    # Final BLEU score
    bleu = bp * geo_mean

    return bleu