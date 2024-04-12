#!/usr/bin/env python3
"""
This script defines a function to add two arrays element-wise.
"""

# Importing the matrix_shape function from 2-size_me_please.py
matrix_shape = __import__('2-size_me_please').matrix_shape

def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise and returns the result.

    Parameters:
    - arr1 (list): The first array.
    - arr2 (list): The second array.

    Returns:
    - list: The resulting array after adding arr1 and arr2 element-wise.
      Returns None if the arrays have different shapes.
    """
    # Check if the arrays have the same shape
    if matrix_shape(arr1) != matrix_shape(arr2):
        return None
    
    return list(map(lambda x, y: x + y, arr1, arr2))
