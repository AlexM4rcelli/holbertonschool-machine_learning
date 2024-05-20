#!/usr/bin/env python3
"""
Task 9
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.
    """
    v_updated = beta1 * v + (1 - beta1) * grad
    s_updated = beta2 * s + (1 - beta2) * np.square(grad)
    v_corrected = v_updated / (1 - np.power(beta1, t))
    s_corrected = s_updated / (1 - np.power(beta2, t))
    var_updated = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var_updated, v_updated, s_updated
