#!/usr/bin/env python3
"""
Task 6
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with tensorflow.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
