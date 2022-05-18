
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


def acc_and_f1(preds: List[Union[int, str]], golds: List[Union[int, str]], prefix=None):
    """
    Computes accuracy, precision, recall and f1 scores for predictions of classification tasks.

    Args:
        preds (List[Union[int, str]]):
            List of predictions, elements can be either integer or string.
        golds (List[Union[int, str]]):
             List of golds, elements can be either integer or string.
        prefix (str, optional):
            Metric name prefix of results.

    Returns:
        results (dict):
            A dict mapping from metric names from metric scores,
            metrics include accuracy (acc), precision (p), recall (r) and f1.
    """
    if not isinstance(preds[0], type(golds[0])):
        logger.warning(f"The element types of golds ({type(golds[0])}) and predictions ({type(preds[0])}) "
                       f"is different, this will cause invalid evaluation results.")
    acc = accuracy_score(y_true=golds, y_pred=preds)
    f1 = f1_score(y_true=golds, y_pred=preds)
    p = precision_score(y_true=golds, y_pred=preds)
    r = recall_score(y_true=golds, y_pred=preds)
    return {
        f"{prefix}_acc" if prefix else "acc": acc,
        f"{prefix}_precision" if prefix else "precision": p,
        f"{prefix}_recall" if prefix else "recall": r,
        f"{prefix}_f1" if prefix else "f1": f1
    }
