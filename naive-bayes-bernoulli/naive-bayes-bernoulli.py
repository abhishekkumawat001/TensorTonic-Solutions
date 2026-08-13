import numpy as np


def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute unnormalized log posteriors for Bernoulli Naive Bayes.

    Returns:
        log_posteriors: shape (n_test, n_classes)
    """

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    # Sorted unique classes
    classes, class_counts = np.unique(y_train, return_counts=True)

    n_train = X_train.shape[0]
    n_classes = len(classes)

    # Log prior: log P(y)
    log_priors = np.log(class_counts / n_train)

    # Count X_i = 1 for every class and feature
    feature_counts = np.zeros((n_classes, X_train.shape[1]))

    for j, cls in enumerate(classes):
        feature_counts[j] = X_train[y_train == cls].sum(axis=0)

    # Laplace smoothing:
    # theta = (count(x_i=1 in class y) + 1) / (n_y + 2)
    theta = (feature_counts + 1) / (class_counts[:, None] + 2)

    # Log probabilities for x_i = 1 and x_i = 0
    log_theta = np.log(theta)
    log_one_minus_theta = np.log(1 - theta)

    # For each test sample:
    #
    # x_i * log(theta_i)
    # + (1-x_i) * log(1-theta_i)
    #
    # Shape:
    # X_test              -> (n_test, d)
    # log_theta           -> (n_classes, d)
    # resulting likelihood -> (n_test, n_classes)
    log_likelihood = (
        X_test[:, None, :] * log_theta[None, :, :]
        + (1 - X_test[:, None, :]) * log_one_minus_theta[None, :, :]
    ).sum(axis=2)

    # Add log prior
    log_posteriors = log_likelihood + log_priors[None, :]

    return log_posteriors