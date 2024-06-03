#!/usr/bin/env python3
"""
Task 1
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = max((kh - 1) // 2, kh // 2)
    pw = max((kw - 1) // 2, kw // 2)
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    output = np.zeros((m, h, w))
    for x in range(h):
        for y in range(w):
            output[:, x, y] = (kernel * images_padded[:,
                                                      x: x + kh,
                                                      y: y + kw
                                                      ]).sum(axis=(1, 2))
    return output
