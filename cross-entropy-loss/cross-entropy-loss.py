import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
     y_true = array -like of shape (N, )
     y_pred = array like of shape (N, C)
         predicted probability of each class 
         return:
         float 
         Average cross entropy loss 
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    correct_class_probs = y_pred[np.arange(n), y_true]
    loss = - np.mean(np.log(correct_class_probs))

    return float(loss)
    
    