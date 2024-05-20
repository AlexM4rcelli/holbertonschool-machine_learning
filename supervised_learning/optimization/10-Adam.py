#!/usr/bin/env python3
"""
Task 10
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow.
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1, beta2=beta2,
                                       epsilon=epsilon)
    return optimizer
