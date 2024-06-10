#!/usr/bin/env python3
"""
Task 3
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.
    """
    m, h_new, w_new, c_new = dA.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            for n in range(c_new):
                if mode == 'max':
                    slice_A_prev = A_prev[:, i * sh: i * sh + kh,
                                          j * sw: j * sw + kw, n]
                    max_val = np.max(slice_A_prev, axis=(1, 2))
                    mask = (slice_A_prev == max_val)[:, :, :, None]
                    dA_val = dA[:, i, j, n][:, None, None, None]
                    dA_prev[:, i * sh: i * sh + kh,
                            j * sw: j * sw + kw, n] += mask * dA_val
                elif mode == 'avg':
                    avg_val = dA[:, i, j, n][:, None, None, None] / (kh * kw)
                    dA_prev[:, i * sh: i * sh + kh,
                            j * sw: j * sw + kw, n] += avg_val

    return dA_prev
