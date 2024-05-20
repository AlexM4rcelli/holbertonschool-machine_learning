#!/usr/bin/env python3
"""
Task 6
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with tensorflow.
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer
