#!/usr/bin/env python3
"""
This script defines a function to determine the shape of a matrix.
"""


def matrix_shape(matrix):
    """
    Determine the shape of a matrix.

    Parameters:
    - matrix: The matrix for which the shape needs to be determined.

    Returns:
    - A list representing the shape of the matrix. For example, for a 2D
    matrix, the shape will be [rows, columns].
    """
    shape = []

    # Check if the matrix is not empty
    if value := matrix:
        shape.append(len(value))

        # Traverse the matrix to determine the dimensions
        while value[0] and isinstance(value[0], list):
            shape.append(len(value[0]))
            value = value[0]

    return shape
