#!/usr/bin/env python3
"""
Task 8
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in tensorflow
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            rho=beta2, epsilon=epsilon)
    return optimizer
