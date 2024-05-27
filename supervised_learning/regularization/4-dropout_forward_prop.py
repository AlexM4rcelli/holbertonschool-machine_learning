#!/usr/bin/env python3
"""
Task 4
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that uses dropout in a forward propagation
    """
    cache = {}
    cache['A0'] = X
    for layer in range(L):
        W = weights["W" + str(layer + 1)]
        A = cache["A" + str(layer)]
        B = weights["b" + str(layer + 1)]
        Z = np.matmul(W, A) + B
        dropout = np.random.rand(Z.shape[0], Z.shape[1])
        dropout = np.where(dropout < keep_prob, 1, 0)
        if layer == L - 1:
            softmax = np.exp(Z)
            cache["A" + str(layer + 1)] = (softmax / np.sum(softmax, axis=0,
                                                            keepdims=True))
        else:
            tanh = np.tanh(Z)
            cache["A" + str(layer + 1)] = tanh
            cache["D" + str(layer + 1)] = dropout
            cache["A" + str(layer + 1)] *= dropout
            cache["A" + str(layer + 1)] /= keep_prob
    return cache