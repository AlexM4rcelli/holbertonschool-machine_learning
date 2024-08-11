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

	if len(poly) == 0:
		return None

	result = [C]

	for i, coef in enumerate(poly):
		integral_coeff = coef / (i + 1)
		if integral_coeff.is_integer():
			integral_coeff = int(integral_coeff)
		result.append(integral_coeff)

	while len(result) > 1 and result[-1] == 0:
		result.pop()

	return result
