#!/usr/bin/env python3
"""
Calculate the sum of squares of integers from 1 to n.
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares of integers from 1 to n.
    """
    sum = 0
    for i in range(1, n + 1):
      sum += i ** 2
    return sum
