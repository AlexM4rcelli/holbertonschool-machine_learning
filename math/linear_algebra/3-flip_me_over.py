#!/usr/bin/env python3
"""
This script defines a function to transpose a matrix.
"""

def matrix_transpose(matrix):
    """
    Transposes a matrix.

    Parameters:
    - matrix (list of lists): The matrix to be transposed.

    Returns:
    - list of lists: The transposed matrix.
    """
    transpose = []

    for i in range(0, len(matrix[0])):
        transpose += [[row[i] for row in matrix]]

    return transpose
