#!/usr/bin/env python3
"""
Task 0
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a
    neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0

    output_h = int((h_prev + 2 * ph - kh) / sh) + 1
    output_w = int((w_prev + 2 * pw - kw) / sw) + 1

    output = np.zeros((m, output_h, output_w, c_new))

    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    for i in range(output_h):
        for j in range(output_w):
            for n in range(c_new):
                output[:, i, j, n] = activation(
                    (A_prev[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :] *
                     W[:, :, :, n]).sum(axis=(1, 2, 3)) + b[0, 0, 0, n]
                )

    return output
