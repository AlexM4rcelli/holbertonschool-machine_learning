#!/usr/bin/env python3
"""
Calculate the sum of squares of integers from 1 to n.
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares of integers from 1 to n.
    """
    if type(n) is not int:
        return None
    elif n < 1:
        return None
    else:
        return int(n * (2 * n + 1))
