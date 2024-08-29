#!/usr/bin/env python3
"""
    Class Neuron
"""
import numpy as np


class Neuron:
    """
        Class Neuron
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(nx).reshape(1, nx)
        self.b = 0
        self.A = 0

    @property
    def W(self):
        """"Return the weights attribute """
        return self.__W

    @property
    def b(self):
        """Return the bias attribute """
        return self.__b

    @property
    def A(self):
        """Return the activation attribute"""
        return self.__A


    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, X) + self.__b)))
        return self.__A
