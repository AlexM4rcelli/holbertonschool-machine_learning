#!/usr/bin/env python3
"""
This script defines a function for matrix multiplication using NumPy.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication using NumPy's @ operator.

    Parameters:
    - mat1: The first matrix as a NumPy array.
    - mat2: The second matrix as a NumPy array.

    Returns:
    - The result of matrix multiplication as a NumPy array.
    """
    return mat1 @ mat2
