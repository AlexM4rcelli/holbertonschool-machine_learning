#!/usr/bin/env python3
"""
Task 4
"""

import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.nn.conv2d(x, filters=6, kernel_size=5, padding='SAME',
                         activation='relu', kernel_initializer=init)
    pool1 = tf.nn.max_pool2d(conv1, ksize=2, strides=2, padding='VALID')

    conv2 = tf.nn.conv2d(pool1, filters=16, kernel_size=5, padding='VALID',
                         activation='relu', kernel_initializer=init)
    pool2 = tf.nn.max_pool2d(conv2, ksize=2, strides=2, padding='VALID')

    flat = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(flat, units=120, activation='relu',
                          kernel_initializer=init)
    fc2 = tf.layers.dense(fc1, units=84, activation='relu',
                          kernel_initializer=init)

    softmax = tf.layers.dense(fc2, units=10, activation='softmax',
                              kernel_initializer=init)

    loss = tf.losses.softmax_cross_entropy(y, softmax)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return softmax, train_op, loss, accuracy
