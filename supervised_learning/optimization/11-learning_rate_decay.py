#!/usr/bin/env python3
"""
Task 11
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.
    """
    decay_factor = 1 / (1 + decay_rate * np.floor(global_step / decay_step))
    alpha_updated = alpha * decay_factor
    return alpha_updated
