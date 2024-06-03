#!/usr/bin/env python3
"""
Task 5
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    output_h = int((h + 2 * ph - kh) / sh) + 1
    output_w = int((w + 2 * pw - kw) / sw) + 1

    output = np.zeros((m, output_h, output_w, nc))

    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    for i in range(output_h):
        for j in range(output_w):
            for n in range(nc):
                output[:, i, j, n] = (
                    images[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :] *
                    kernels[:, :, :, n]
                ).sum(axis=(1, 2, 3))

    return output
