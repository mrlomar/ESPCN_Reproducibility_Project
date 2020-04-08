import os
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt
import torch
from skimage.transform import *
from skimage.filters import *

from src.espcn import PS


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def average_PSNR(folder, net, r, gaussianSigma):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder, filename))
        if img is not None:
            img = resize(img, ((img.shape[0] // r) * r, (img.shape[1] // r) * r))
            images.append(img)

    sumPSNR = 0
    for og_img in images:
        img_blurred = gaussian(og_img, sigma=gaussianSigma,
                               multichannel=True)  # multichannel blurr so that 3rd channel is not blurred
        img = resize(img_blurred, (img_blurred.shape[0] // r, img_blurred.shape[1] // r))
        if (len(img.shape) == 2):  # convert image to rgb if it is grayscale
            img = np.stack((img, img, img), axis=2)
            og_img = np.stack((og_img, og_img, og_img), axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img).unsqueeze(0).double()
        result = net(img).detach().numpy()
        sumPSNR += PSNR(PS(result[0], r) * 255, og_img * 255)

    return sumPSNR / len(images)
