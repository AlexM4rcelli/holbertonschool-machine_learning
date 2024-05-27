#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    """
    l2_cost = 0
    for key, values in weights.items():
        if key[0] == 'W':
            l2_cost += np.linalg.norm(values)
    l2_cost *= (lambtha / (2 * m))
    cost_reg = cost + l2_cost
    return cost_reg
