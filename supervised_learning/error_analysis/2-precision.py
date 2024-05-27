#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    """
    positivies = np.diag(confusion)
    negatives = np.sum(confusion, axis=0) - positivies
    precision = positivies / (positivies + negatives)
    return precision
