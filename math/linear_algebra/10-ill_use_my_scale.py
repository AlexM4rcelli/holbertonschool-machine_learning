#!/usr/bin/env python3
"""
This script defines a function to get the shape of a matrix using NumPy.
"""


def np_shape(matrix):
    """
    Get the shape of a matrix using NumPy's shape attribute.

    Parameters:
    - matrix: The matrix whose shape needs to be determined, as a NumPy array.

    Returns:
    - A tuple representing the shape of the matrix, where the first element
    is the number of rows
      and the second element is the number of columns.
    """
    return matrix.shape
