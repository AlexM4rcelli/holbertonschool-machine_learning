#!/usr/bin/env python3
"""
Task 6
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Creates a stack bar plot
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    names = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    plt.figure(figsize=(6.4, 4.8))
    bottom_positions = np.zeros(len(names))
    for i in range(fruit.shape[0]):
        plt.bar(names, fruit[i], bottom=bottom_positions,
                width=0.5, color=colors[i], label=labels[i]
                )
        bottom_positions += fruit[i]
    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.legend(loc='upper right')
    plt.ylim(0, 80)
    plt.show()
