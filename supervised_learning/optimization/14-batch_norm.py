#!/usr/bin/env python3
"""
Task 14
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    base_layer = tf.keras.layers.Dense(n, kernel_initializer=initializer)
    Z = base_layer(prev)
    batch_norm_layer = tf.keras.layers.BatchNormalization(epsilon=1e-7)
    Z_norm = batch_norm_layer(Z, training=True)
    a = activation(Z_norm)
    return a
