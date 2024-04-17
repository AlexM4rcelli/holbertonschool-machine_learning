#!/usr/bin/env python3
"""
This script defines a function for element-wise operations on matrices using
NumPy.
"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication,
    and division of two matrices.

    Parameters:
    - mat1: The first matrix as a NumPy array.
    - mat2: The second matrix as a NumPy array.

    Returns:
    - A list containing the results of element-wise addition, subtraction,
    multiplication, and division,
      in the order [mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2].
    """
    return [mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2]
