#!/usr/bin/env python3
"""
This module defines a function for concatenating two matrices
along a specified axis.
"""


def matrix_shape(matrix):
    shape = []
    if value := matrix:
        shape.append(len(value))

        while isinstance(value[0], list) and value[0]:
            shape.append(len(value[0]))
            value = value[0]

    return shape


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
    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)

    if shape_mat1[0] < shape_mat2[0]:
        return None

    if axis == 0:
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]
    if axis == 1:
        cated = []
        for r in range(len(mat2.copy())):
            new = []
            for c in range(len(mat2.copy()[r])):
                new.append(mat2[r].copy()[c])
            cated.append(mat1[r].copy() + new)
        return cated

    return None
