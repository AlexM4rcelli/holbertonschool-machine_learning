#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization.
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for lay in range(L, 0, -1):
        dw = (1 / m) * np.dot(dz, cache['A' + str(lay - 1)].T) +\
            ((lambtha / m) * weights['W' + str(lay)])
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        if lay > 1:
            dz = np.dot(weights['W' + str(lay)].T, dz) *\
                (1 - cache['A' + str(lay - 1)] ** 2)
        weights['W' + str(lay)] -= alpha * dw
        weights['b' + str(lay)] -= alpha * db
