#!/usr/bin/env python3
"""
Task 6
"""


import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation, \
        kernel_initializer=initializer)
    drop = tf.layers.Dropout(rate=1-keep_prob, training=training)
    output = drop(layer(prev))
    return output
