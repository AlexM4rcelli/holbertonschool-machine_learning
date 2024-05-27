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
    for l in range(L, 0, -1):
        dw = (1 / m) * np.dot(dz, cache['A' + str(l - 1)].T) +\
            ((lambtha / m) * weights['W' + str(l)])
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        if l > 1:
            dz = np.dot(weights['W' + str(l)].T, dz) *\
                (1 - cache['A' + str(l - 1)] ** 2)
        weights['W' + str(l)] -= alpha * dw
        weights['b' + str(l)] -= alpha * db
