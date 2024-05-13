#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
	def __init__(self, nx, layers, activation='sig'):
		if type(nx) is not int:
			raise TypeError("nx must be an integer")
		if nx < 1:
			raise ValueError("nx must be a positive integer")
		if type(layers) is not list or not all(map(lambda x: type(x) is int and x > 0, layers)):
			raise TypeError("layers must be a list of positive integers")
		if activation not in ['sig', 'tanh']:
			raise ValueError("activation must be 'sig' or 'tanh'")

		self.__activation = activation
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

	@property
	def activation(self):
		return self.__activation

	def forward_prop(self, X):
		self.__cache["A0"] = X
		for i in range(self.__L):
			Z = np.matmul(self.__weights["W" + str(i + 1)], self.__cache["A" + str(i)]) + self.__weights["b" + str(i + 1)]
			if i == self.__L - 1:
				t = np.exp(Z)
				self.__cache["A" + str(i + 1)] = t / np.sum(t, axis=0, keepdims=True)
			else:
				if self.__activation == 'sig':
					self.__cache["A" + str(i + 1)] = 1 / (1 + np.exp(-Z))
				elif self.__activation == 'tanh':
					self.__cache["A" + str(i + 1)] = np.tanh(Z)
		return self.__cache["A" + str(self.__L)], self.__cache


	def cost(self, Y, A):
		m = Y.shape[1]
		cost = -1 / m * np.sum(Y * np.log(A))
		return cost

	def evaluate(self, X, Y):
		A = self.forward_prop(X)[0]
		cost = self.cost(Y, A)
		prediction = np.argmax(A, axis=0)
		label = np.argmax(Y, axis=0)
		accuracy = np.sum(prediction == label) / prediction.size
		return prediction, accuracy, cost

	def gradient_descent(self, Y, cache, alpha=0.05):
		m = Y.shape[1]
		dZ = cache["A" + str(self.__L)] - Y
		for i in range(self.__L, 0, -1):
			A_prev = cache["A" + str(i - 1)]
			dW = np.matmul(dZ, A_prev.T) / m
			db = np.sum(dZ, axis=1, keepdims=True) / m
			if self.__activation == 'sig':
				dZ = np.matmul(self.__weights["W" + str(i)].T, dZ) * (A_prev * (1 - A_prev))
			elif self.__activation == 'tanh':
				dZ = np.matmul(self.__weights["W" + str(i)].T, dZ) * (1 - A_prev**2)
			self.__weights["W" + str(i)] -= alpha * dW
			self.__weights["b" + str(i)] -= alpha * db

	def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
		if type(iterations) is not int:
			raise TypeError("iterations must be an integer")
		if iterations <= 0:
			raise ValueError("iterations must be a positive integer")
		if type(alpha) is not float:
			raise TypeError("alpha must be a float")
		if alpha <= 0:
			raise ValueError("alpha must be positive")
		if verbose is True or graph is True:
			if type(step) is not int:
				raise TypeError("step must be an integer")
			if step <= 0 or step > iterations:
				raise ValueError("step must be positive and <= iterations")

		costs = []
		for i in range(iterations):
			self.forward_prop(X)
			cost = self.cost(Y, self.__cache["A" + str(self.__L)])
			if verbose and i % step == 0:
				print("Cost after {} iterations: {}".format(i, cost))
			if i % step == 0 or i == iterations:
				costs.append(cost)
			self.gradient_descent(Y, self.__cache, alpha)

		if graph:
			plt.plot(np.arange(0, iterations + 1, step), costs)
			plt.xlabel('iteration')
			plt.ylabel('cost')
			plt.title('Training Cost')
			plt.show()

		return self.evaluate(X, Y)

	def save(self, filename):
		if not filename.endswith('.pkl'):
			filename += '.pkl'
		with open(filename, 'wb') as file:
			pickle.dump(self, file)
   
	@staticmethod
	def load(filename):
		try:
			with open(filename, 'rb') as file:
				return pickle.load(file)
		except FileNotFoundError:
			return None
