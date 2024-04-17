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
    plt.hist(student_grades, edgecolor='black')

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()
