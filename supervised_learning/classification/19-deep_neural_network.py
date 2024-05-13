#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
	def __init__(self, nx, layers):
		if type(nx) is not int:
			raise TypeError("nx must be an integer")
		if nx < 1:
			raise ValueError("nx must be a positive integer")
		if type(layers) is not list or not all(map(lambda x: type(x) is int and x > 0, layers)):
			raise TypeError("layers must be a list of positive integers")

		self.__L = len(layers)
		self.__cache = {}
		self.__weights = {}
		for i in range(self.__L):
			if i == 0:
				self.__weights["W" + str(i + 1)] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
			else:
				self.__weights["W" + str(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
			self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

	@property
	def L(self):
		return self.__L

	@property
	def cache(self):
		return self.__cache

	@property
	def weights(self):
		return self.__weights

	def forward_prop(self, X):
		self.__cache["A0"] = X
		for i in range(self.__L):
			Z = np.dot(self.__weights["W" + str(i + 1)], self.__cache["A" + str(i)]) + self.__weights["b" + str(i + 1)]
			self.__cache["A" + str(i + 1)] = 1 / (1 + np.exp(-Z))
		return self.__cache["A" + str(self.__L)], self.__cache

	def cost(self, Y, A):
		m = Y.shape[1]
		return -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

