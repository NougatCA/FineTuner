#!/usr/bin/env python
#
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np


def my_lcs(string, sub):
    """
    Calculates the longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(0, len(sub) + 1)] for _ in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


# class Rouge(object):
#     """
#     Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
#     """

def calc_score(candidate, refs, beta=1.2):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :param beta:
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    assert (len(candidate) == 1)
    assert (len(refs) > 0)
    prec = []
    rec = []

    # split into tokens
    token_c = candidate[0].split(" ")

    for reference in refs:
        # split into tokens
        token_r = reference.split(" ")
        # compute the longest common subsequence
        lcs = my_lcs(token_r, token_c)
        prec.append(lcs / float(len(token_c)))
        rec.append(lcs / float(len(token_r)))

    prec_max = max(prec)
    rec_max = max(rec)

    if prec_max != 0 and rec_max != 0:
        score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
    else:
        score = 0.0
    return score


def rouge_l(preds: list[str], golds: list[str], prefix=None):

    scores = []

    for pred, gold in zip(preds, golds):
        pred = pred.strip().split()
        gold = gold.strip().split()
        # if len(pred) * len(gold) == 0:
        #     score = 1 if len(pred) == len(gold) else 0
        # else:
        score = calc_score(candidate=[" ".join(pred)], refs=[" ".join(gold)])
        scores.append(score)

    avg_score = np.mean(scores)
    return {f"{prefix}_rouge_l" if prefix else "rouge_l": avg_score * 100}
