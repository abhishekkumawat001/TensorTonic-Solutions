import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)

    # Build confusion matrix
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[class_to_index[t], class_to_index[p]] += 1

    # Accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    tp_list = []
    fp_list = []
    fn_list = []
    support_list = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])

        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        support_list.append(support)

    tp_list = np.array(tp_list)
    fp_list = np.array(fp_list)
    fn_list = np.array(fn_list)
    support_list = np.array(support_list)

    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    if average == "micro":
        TP = np.sum(tp_list)
        FP = np.sum(fp_list)
        FN = np.sum(fn_list)

        precision = safe_div(TP, TP + FP)
        recall = safe_div(TP, TP + FN)
        f1 = safe_div(2 * precision * recall, precision + recall)

    elif average == "macro":
        precisions = []
        recalls = []
        f1s = []

        for tp, fp, fn in zip(tp_list, fp_list, fn_list):
            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            f = safe_div(2 * p * r, p + r)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

    elif average == "weighted":
        precisions = []
        recalls = []
        f1s = []

        for tp, fp, fn in zip(tp_list, fp_list, fn_list):
            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            f = safe_div(2 * p * r, p + r)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        weights = support_list / np.sum(support_list)

        precision = np.sum(np.array(precisions) * weights)
        recall = np.sum(np.array(recalls) * weights)
        f1 = np.sum(np.array(f1s) * weights)

    elif average == "binary":
        if pos_label not in class_to_index:
            precision = recall = f1 = 0.0
        else:
            i = class_to_index[pos_label]
            tp = tp_list[i]
            fp = fp_list[i]
            fn = fn_list[i]

            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall)

    else:
        raise ValueError("average must be 'micro', 'macro', 'weighted', or 'binary'")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }