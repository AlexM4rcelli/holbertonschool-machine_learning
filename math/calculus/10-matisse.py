#!/usr/bin/env python3
"""
Calculate the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.
    """
    if not isinstance(poly, list):
        return None
    if not all(isinstance(coeff, int) for coeff in poly):
        return None

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])
    return derivative
