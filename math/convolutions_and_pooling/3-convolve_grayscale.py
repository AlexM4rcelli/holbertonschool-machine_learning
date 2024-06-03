#!/usr/bin/env python3
"""
Task 3
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Function to perform a grayscale convolution
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
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

    output = np.zeros((m, output_h, output_w))

    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (images[:, i * sh: i * sh + kh,
                                      j * sw: j * sw + kw] * kernel
                               ).sum(axis=(1, 2))

    return output
