#!/usr/bin/env python3
"""
Task 5
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable
    """
    v_updated = beta1 * v + (1 - beta1) * grad
    var_updated = var - alpha * v_updated
    return var_updated, v_updated
