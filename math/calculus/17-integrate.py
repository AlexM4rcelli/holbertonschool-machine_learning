#!/usr/bin/env python3
"""
	Calculate the indefinite integral of a polynomial.
"""


def poly_integral(poly, C=0):
	"""
	Calculate the indefinite integral of a polynomial.
	"""
	if not isinstance(poly, list):
		return None
	elif not all(isinstance(coeff, int) for coeff in poly):
		return None
	elif not isinstance(C, int):
		return None

	result = [C]

	if len(poly) > 1:
		for i in range(len(poly)):
			integral_coeff = poly[i] / (i + 1)
			if integral_coeff.is_integer():
				integral_coeff = int(integral_coeff)
			result.append(integral_coeff)
		return result
	else:
		return result + poly
