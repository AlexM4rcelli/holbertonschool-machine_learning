#!/usr/bin/env python3
"""
This script defines a function to add two 2D matrices.
"""

matrix_shape = __import__('2-size_me_please').matrix_shape

def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise and returns the result.

    Parameters:
    - mat1 (list of lists): The first 2D matrix.
    - mat2 (list of lists): The second 2D matrix.

    Returns:
    - list of lists: The resulting 2D matrix after adding mat1 and mat2 element-wise.
      Returns None if the matrices have different shapes.
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    
    summ = []
    for c in range(0, len(mat1)):
        new = []
        for r in range(0, len(mat1[c])):
            new.append(mat1[c][r] + mat2[c][r])
        summ += [new]
    return summ
