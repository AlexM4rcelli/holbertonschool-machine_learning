"""
This module defines a function for matrix multiplication using matrix shapes.
"""

matrix_shape = __import__('2-size_me_please').matrix_shape

def mat_mul(mat1, mat2):
    """
    Multiply two matrices and return the result if they are compatible for multiplication.

    Parameters:
    - mat1: The first matrix as a list of lists.
    - mat2: The second matrix as a list of lists.

    Returns:
    - The result of the matrix multiplication as a new matrix if mat1 and mat2 are compatible.
    - None if mat1 and mat2 cannot be multiplied due to incompatible dimensions.
    """
    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)

    if shape_mat1[1] == shape_mat2[0]:
        res = []

        for i in range(len(mat1)):
            new = []
            for j in range(len(mat2[0])):
                summable = []
                for k in range(len(mat2)):
                    summable.append(mat1[i][k] * mat2[k][j])
                new.append(sum(summable))
            res += [new]

        return res

    return None
