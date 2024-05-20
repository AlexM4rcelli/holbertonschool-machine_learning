#!/usr/bin/env python3
"""
Task 13
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    """
    Z_mean = np.mean(Z, axis=0)
    Z_var = np.var(Z, axis=0)
    Z_norm = (Z - Z_mean) / np.sqrt(Z_var + epsilon)
    Z_norm = gamma * Z_norm + beta
    return Z_norm
