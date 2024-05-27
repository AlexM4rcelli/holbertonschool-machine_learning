#!/usr/bin/env python3
"""
Task 5
"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout
    regularization using gradient descent.
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for lay in range(L, 0, -1):
        dw = (1 / m) * np.dot(dz, cache['A' + str(lay - 1)].T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        if lay > 1:
            dz = np.dot(weights['W' + str(lay)].T, dz) * \
                (1 - cache['A' + str(lay - 1)] ** 2)
            dz *= cache['D' + str(lay - 1)]
            dz /= keep_prob
        weights['W' + str(lay)] -= alpha * dw
        weights['b' + str(lay)] -= alpha * db
