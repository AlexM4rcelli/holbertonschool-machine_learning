#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        m = Y.shape[1]
        return np.sum((Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / -m)

    def evaluate(self, X, Y):
        prediction = np.where(self.forward_prop(X) >= 0.5, 1, 0)
        
        return [prediction, self.cost(Y, self.__A)]

    def gradient_descent(self, X, Y, A, alpha):
        """Calculates one pass of gradient"""
        m = X.shape[1]
        dz = A - Y
        dw = np.dot(dz, X.T) * 1 / m
        db = np.sum(dz) / m
        self.__b = self.__b - alpha * db
        self.__W = self.__W - alpha * dw
    
    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
            graph=True, step=100):
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_data = []
        step_data = []

        cost_data.append(self.cost(Y, self.__A))
        step_data.append(0)
        
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            
            if i % step == 0:
                cost_data.append(cost)
                step_data.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

            self.gradient_descent(X, Y, self.__A, alpha)

        cost_data.append(self.cost(Y, self.__A))
        step_data.append(iterations)

        if graph:
            plt.plot(step_data, cost_data, color='blue')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
