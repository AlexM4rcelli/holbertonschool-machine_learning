#!/usr/bin/env python3
"""
Task 7
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.
    """
    s_updated = beta2 * s + (1 - beta2) * np.square(grad)
    var_updated = var - alpha * grad / (np.sqrt(s_updated) + epsilon)
    return var_updated, s_updated
