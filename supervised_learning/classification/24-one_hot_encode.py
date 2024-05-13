#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Task 24
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Function that converts a numeric label vector into a one-hot matrix
    """
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    if len(Y) == 0 or classes <= np.amax(Y):
        return None
    return np.eye(classes)[Y].T
