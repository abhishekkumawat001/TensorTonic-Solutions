import math

def perplexity(prob_distributions, actual_tokens):
    N = len(actual_tokens)
    
    log_sum = 0.0
    
    for i in range(N):
        p = prob_distributions[i][actual_tokens[i]]
        log_sum += math.log(p)
    
    # Cross-entropy
    H = - (1 / N) * log_sum
    
    # Perplexity
    PP = math.exp(H)
    
    return float(PP)