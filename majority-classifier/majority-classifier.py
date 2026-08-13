import numpy as np


def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """

    y_train = np.asarray(y_train)

    # Find unique classes and their frequencies
    classes, counts = np.unique(y_train, return_counts=True)

    # Class with highest frequency
    majority_class = classes[np.argmax(counts)]

    # Create prediction for every test sample
    predictions = np.full(len(X_test), majority_class, dtype=int)

    return predictions