import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)


def score_fn(gold: str, pred: str):
    """
    Custom exact-match score. Returns 1.0 if strings match (case-insensitive), 0.0 otherwise.
    """
    gold_norm = str(gold).strip().upper()
    pred_norm = str(pred).strip().upper()
    return 1.0 if gold_norm == pred_norm else 0.0


def score_fn_vectorized(golds, preds):
    """
    Vectorized version of score_fn for numpy arrays.
    """
    golds_arr = np.array(golds, dtype=str)
    preds_arr = np.array(preds, dtype=str)

    golds_norm = np.char.upper(np.char.strip(golds_arr))
    preds_norm = np.char.upper(np.char.strip(preds_arr))

    return (golds_norm == preds_norm).astype(float)


def compute_overall_metrics(y_true, y_pred, label_encoder=None):
    """
    Computes standard classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
    }
