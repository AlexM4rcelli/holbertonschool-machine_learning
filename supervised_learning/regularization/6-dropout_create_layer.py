#!/usr/bin/env python3
"""
Task 6
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    drop = tf.layers.Dropout(keep_prob, training=training)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=drop)
    output = drop(layer(prev))
    return output
