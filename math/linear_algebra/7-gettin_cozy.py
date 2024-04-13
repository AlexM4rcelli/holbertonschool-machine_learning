#!/usr/bin/env python3
"""
This module defines a function for concatenating two matrices
along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specified axis and return the result if
    they are compatible for concatenation.

    Parameters:
    - mat1: The first matrix as a list of lists.
    - mat2: The second matrix as a list of lists.
    - axis: The axis along which to concatenate the matrices.
    Default is 0 (concatenate along rows).

    Returns:
    - The result of concatenating mat1 and mat2 along the specified axis
    if they are compatible.
    - None if mat1 and mat2 cannot be concatenated due to incompatible
    dimensions.
    """

    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    elif axis == 1 and len(mat1) != len(mat2):
        return None

    if axis == 0:
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]
    if axis == 1:
        return [mat1[r][:] + mat2[r][:] for r in range(len(mat1))]

    return None
