#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.
    """

    sens = sensitivity(confusion)
    prec = precision(confusion)

    f1_score = 2 * ((prec * sens) / (prec + sens))
    return f1_score
