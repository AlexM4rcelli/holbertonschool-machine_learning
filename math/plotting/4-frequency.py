#!/usr/bin/env python3
"""
Task 4
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Creates a histogram
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades, edgecolor='black', bins=range(0, 101, 10))
    plt.xticks(np.arange(0, 101, 10))
    plt.xlabel('Grades')
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()
