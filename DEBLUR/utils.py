
import torch
import numpy as np
import math

# input type:tensor ; output type:tensor


def add_noise(input_img, noise_sigma):
    noise_sigma = noise_sigma / 255
    noise_img = torch.clamp(input_img+noise_sigma*torch.randn_like(input_img), 0.0, 1.0)

    return noise_img


def PSNR(img1, img2, color=False):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return mse * 255 * 255, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

















