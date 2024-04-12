#!/usr/bin/env python3
"""
This script defines a function for concatenating matrices along a
specified axis using NumPy.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate matrices along a specified axis using NumPy's concatenate
    function.

    Parameters:
    - mat1: The first matrix as a NumPy array.
    - mat2: The second matrix as a NumPy array.
    - axis: The axis along which to concatenate the matrices. Default is 0
    (concatenate along rows).

    Returns:
    - The concatenated matrix as a NumPy array.
    """
    return np.concatenate((mat1, mat2), axis)
