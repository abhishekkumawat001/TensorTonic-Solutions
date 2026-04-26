import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Write code here
    real_scores = np.asarray(real_scores)
    fake_scores = np.asarray(fake_scores)
    # real_scores = np.ndarray(real_scores)
    # fake_scores = np.ndarray(fake_scores)

    loss = - np.mean(real_scores) + np.mean(fake_scores)

    return float(loss)