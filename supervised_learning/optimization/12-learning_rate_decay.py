#!/usr/bin/env python3
"""
Task 12
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using inverse
    time decay.
    """
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(alpha,
                                                global_step,
                                                decay_step,
                                                decay_rate,
                                                staircase=True)
    return learning_rate
