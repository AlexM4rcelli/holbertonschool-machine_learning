import numpy as np


class Neuron:
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
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
