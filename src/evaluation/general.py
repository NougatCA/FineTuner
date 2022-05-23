
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import List, Union, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def acc(preds: List[Union[int, str]], golds: List[Union[int, str]], prefix=None) -> dict:
    """
        Computes accuracy for predictions of classification tasks.

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
                metrics include accuracy (acc).
        """
    if not isinstance(preds[0], type(golds[0])):
        logger.warning(f"The element types of golds ({type(golds[0])}) and predictions ({type(preds[0])}) "
                       f"is different, this will cause invalid evaluation results.")
    acc = accuracy_score(y_true=golds, y_pred=preds)
    return {
        f"{prefix}_acc" if prefix else "acc": acc * 100,
    }


def p_r_f1(preds: List[Union[int, str]], golds: List[Union[int, str]], prefix=None, pos_label=1) -> dict:
    """
    Computes precision, recall and f1 scores for predictions of classification tasks.

    Args:
        preds (List[Union[int, str]]):
            List of predictions, elements can be either integer or string.
        golds (List[Union[int, str]]):
             List of golds, elements can be either integer or string.
        prefix (str, optional):
            Metric name prefix of results.
        pos_label (optional, defaults to 1):
            The positive label, defaults to 1.

    Returns:
        results (dict):
            A dict mapping from metric names from metric scores,
            metrics include precision (p), recall (r) and f1.
    """
    if not isinstance(preds[0], type(golds[0])):
        logger.warning(f"The element types of golds ({type(golds[0])}) and predictions ({type(preds[0])}) "
                       f"is different, this will cause invalid evaluation results.")
    f1 = f1_score(y_true=golds, y_pred=preds, pos_label=pos_label)
    p = precision_score(y_true=golds, y_pred=preds, pos_label=pos_label)
    r = recall_score(y_true=golds, y_pred=preds, pos_label=pos_label)
    return {
        f"{prefix}_precision" if prefix else "precision": p * 100,
        f"{prefix}_recall" if prefix else "recall": r * 100,
        f"{prefix}_f1" if prefix else "f1": f1 * 100
    }


def map_score(scores, sort_ids, labels, prefix=None) -> dict:
    """
    Computes Mean Average Precision (MAP) score for retrieval tasks.

    Args:
        scores (np.ndarray):
            A square metrics representing relevant scores between examples.
        sort_ids (np.ndarray):
            A sorted scores.
        labels (np.ndarray):
            Labels (classes) of examples.
        prefix (str, optional):
            Metric name prefix of results.

    Returns:
        map_value (float):
            MAP score.

    """
    dic = {}
    for i in range(scores.size(0)):
        scores[i, i] = -1000000
        if int(labels[i]) not in dic:
            dic[int(labels[i])] = -1
        dic[int(labels[i])] += 1
    map_scores = []
    for i in range(scores.size(0)):
        label = int(labels[i])
        ap = []
        for j in range(dic[label]):
            index = sort_ids[i, j]
            if int(labels[index]) == label:
                ap.append((len(ap) + 1) / (j + 1))
        map_scores.append(sum(ap) / dic[label])

    return {f"{prefix}_map" if prefix else "map": float(np.mean(map_scores)) * 100}


def mrr(scores, prefix=None) -> dict:

    ranks = []
    for i in range(len(scores)):
        score = scores[i, i]
        rank = 1
        for j in range(len(scores)):
            if i != j and scores[i, j] >= score:
                rank += 1
        ranks.append(1 / rank)

    return {f"{prefix}_mrr" if prefix else "mrr": float(np.mean(ranks)) * 100}


def exact_match(preds, golds, prefix) -> dict:
    assert len(preds) == len(golds)
    count = 0
    for pred, gold in zip(preds, golds):
        if pred == gold:
            count += 1
    avg_score = count / len(preds)
    return {f"{prefix}_em" if prefix else "em": avg_score * 100}
