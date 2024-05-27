#!/usr/bin/env python3
"""
Task 2
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Function to calculate the L2 regularization in tf
    """
    l2_cost = tf.losses.get_regularization_losses()
    return (cost + l2_cost)
