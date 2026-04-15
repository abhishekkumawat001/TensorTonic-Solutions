import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    x = np.asarray(x)
    
    n = len(x)
    
    # Set random generator
    if rng is None:
        rng = np.random.default_rng()
    
    # Bootstrap sampling
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(x, size=n, replace=True)
        boot_means.append(np.mean(sample))
    
    boot_means = np.array(boot_means)
    
    # Confidence interval
    alpha = 1 - ci
    lower = np.quantile(boot_means, alpha / 2)
    upper = np.quantile(boot_means, 1 - alpha / 2)
    
    return boot_means, float(lower), float(upper)