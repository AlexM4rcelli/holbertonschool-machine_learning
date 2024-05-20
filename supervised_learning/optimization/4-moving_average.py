#!/usr/bin/env python3
"""
Task 4
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set.
    """
    averages = []
    v = 0
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        correction = v / (1 - beta**(i + 1))
        averages.append(correction)
    return averages
