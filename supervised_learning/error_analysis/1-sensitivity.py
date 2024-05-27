#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.
    """
    positivies = np.diag(confusion)
    negatives = np.sum(confusion, axis=1) - positivies
    sensitivity = positivies / (positivies + negatives)
    return sensitivity
