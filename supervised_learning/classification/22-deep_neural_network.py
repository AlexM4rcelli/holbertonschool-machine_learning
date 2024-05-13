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
		return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

	def evaluate(self, X, Y):
		A = self.forward_prop(X)[0]
		return np.round(A).astype(int), self.cost(Y, A)

	def gradient_descent(self, Y, cache, alpha=0.05):
		m = Y.shape[1]
		dZ = cache["A" + str(self.__L)] - Y
		for i in range(self.__L, 0, -1):
			A = self.__cache["A{}".format(i - 1)]
			dW = np.dot(dZ, A.T) / m
			db = np.sum(dZ, axis=1, keepdims=True) / m
			dZ = np.dot(self.__weights["W{}".format(i)].T, dZ) * A * (1 - A)
			self.__weights["W" + str(i)] -= alpha * dW
			self.__weights["b" + str(i)] -= alpha * db

	def train(self, X, Y, iterations=5000, alpha=0.05):
		if type(iterations) is not int:
			raise TypeError("iterations must be an integer")
		if iterations <= 0:
			raise ValueError("iterations must be a positive integer")
		if type(alpha) is not float:
			raise TypeError("alpha must be a float")
		if alpha <= 0:
			raise ValueError("alpha must be positive")

		for _ in range(iterations):
			self.forward_prop(X)
			self.gradient_descent(Y, self.__cache, alpha)

		return self.evaluate(X, Y)
