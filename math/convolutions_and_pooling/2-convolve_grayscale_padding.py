#!/usr/bin/env python3
"""
Task 2
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a convolution on grayscale
    images with custom padding
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convoluted_h = h + 2 * ph - kh + 1
    convoluted_w = w + 2 * pw - kw + 1
    convoluted_images = np.zeros((m, convoluted_h, convoluted_w))

    for i in range(convoluted_h):
        for j in range(convoluted_w):
            image_slide = padded_images[:, i: i + kh, j: j + kw]
            convoluted_images[:, i, j] = np.sum(image_slide * kernel,
                                                axis=(1, 2))

    return convoluted_images
