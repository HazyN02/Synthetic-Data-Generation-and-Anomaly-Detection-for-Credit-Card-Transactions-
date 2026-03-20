import numpy as np
from sklearn.metrics import roc_curve, average_precision_score


# ------------------------------------------------------------------
# Metrics used everywhere
# ------------------------------------------------------------------

def recall_at_fpr(y_true, y_score, target_fpr=0.01):
    """
    Recall at a fixed false positive rate.
    This is standard for fraud detection.
    """

    fpr, tpr, _ = roc_curve(y_true, y_score)

    # first index where FPR exceeds target
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)

    return tpr[idx]


def pr_auc(y_true, y_score):
    """
    Precision-Recall AUC (Average Precision)
    """
    return average_precision_score(y_true, y_score)
