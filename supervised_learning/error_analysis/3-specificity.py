#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    true_negatives = np.sum(confusion) - np.sum(confusion, axis=0) - \
        np.sum(confusion, axis=1) + true_positives
    specificity = true_negatives / (true_negatives + false_positives)
    return specificity
