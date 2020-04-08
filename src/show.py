import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.espcn import *
from math import log10, sqrt
import cv2
import numpy as np
from skimage.transform import *
from src.metrics import PSNR
import sys

# hyperparameters
r = 3  # upscaling ratio

# Constants
C = 3  # colour channels
use_gpu = torch.cuda.is_available()

# Retrieve model folder
if len(sys.argv) < 2:
	raise Exception('No model_folder was given as system argument! Please use "python show.py model_folder"')

model_folder = sys.argv[1]

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

net = Net(r, C)
net.double()
net.load_state_dict(torch.load("../models/" + model_folder + "/trained_model"))
net.eval()

first_img = plt.imread("../datasets/testing/Set14/baboon.png")
first_img = resize(first_img, (480, 498))

png = Image.fromarray((first_img * 255).round().astype(np.uint8))
png.save("HR.png")

plt.imshow(first_img)
plt.show()

img = resize(first_img, (first_img.shape[0] // r, first_img.shape[1] // r))

png = Image.fromarray((img * 255).round().astype(np.uint8))
png.save("LR.png")

plt.imshow(img)
plt.show()

img = np.transpose(img, (2, 0, 1))
img = torch.Tensor(img).unsqueeze(0).double()
net.cpu()
result = net(img).detach().numpy()
result = np.clip(PS(result[0], r), 0, 1)


plt.imshow(result)
plt.show()

png = Image.fromarray((result * 255).round().astype(np.uint8))
png.save("SR.png")

train_losses = np.load("../models/" + model_folder + "/loss_train.npy")
test_losses = np.load("../models/" + model_folder + "/loss_test.npy")

plt.plot(train_losses)
plt.plot(test_losses)
plt.yscale("log")
plt.show()

print("PSNR:" + str(PSNR(result * 255, first_img * 255)))