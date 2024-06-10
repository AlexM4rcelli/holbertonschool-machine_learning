#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0

    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(h_new):
        for j in range(w_new):
            for n in range(c_new):
                slice_A_prev = A_prev[:, i * sh: i * sh + kh,
                                      j * sw: j * sw + kw, :]
                dZ_val = dZ[:, i, j, n][:, None, None, None]
                dA_prev[:, i * sh: i * sh + kh,
                        j * sw: j * sw + kw, :] += W[:, :, :, n] * dZ_val
                dW[:, :, :, n] += (slice_A_prev * dZ_val).sum(axis=0)
                db[:, :, :, n] += dZ[:, i, j, n].sum()

    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
