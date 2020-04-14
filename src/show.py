import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.espcn import *
from math import log10, sqrt
import cv2
import numpy as np
from skimage.transform import *
from src.metrics import PSNR, average_PSNR
import sys
import csv

from skimage.filters import *

gaussianSigma = 0.1  # gaussian sigma used when downscaling

# Constants
C = 3  # colour channels
use_gpu = torch.cuda.is_available()

# Retrieve model folder
if len(sys.argv) < 2:
    raise Exception('No model_folder was given as system argument! Please use "python show.py model_folder"')

model_folder = sys.argv[1]
with open("../models/" + model_folder + "/results.csv") as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)
    headers = rows[0]
    r = eval(rows[2][headers.index("r")])
    blur = eval(rows[2][headers.index("blur")])

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


net = Net(r, C)
net.double()
net.load_state_dict(torch.load("../models/" + model_folder + "/best_model"))
net.eval()

first_img = plt.imread("../datasets/testing/Set14/baboon.png")
first_img = resize(first_img, (480, 498))

png = Image.fromarray((first_img * 255).round().astype(np.uint8))
png.save("HR.png")

plt.imshow(first_img)
plt.show()

img_blurred = gaussian(first_img, sigma=gaussianSigma,
                       multichannel=True)  # multichannel blurr so that 3rd channel is not blurred
img = resize(img_blurred, (img_blurred.shape[0] // r, img_blurred.shape[1] // r))

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

print("PSNR:" + str(PSNR(result * 255, first_img * 255)))

train_losses = np.load("../models/" + model_folder + "/loss_train.npy")
test_losses = np.load("../models/" + model_folder + "/loss_test.npy")

plt.plot(train_losses[100:-1], label="train losses")
plt.plot(test_losses[100:-1], label="test losses")
plt.yscale("log")
plt.legend()
plt.show()

set5_PSNR = average_PSNR("../datasets/testing/Set5", net, r, blur)
print("PSNR Set5:   ", set5_PSNR)
set14_PSNR = average_PSNR("../datasets/testing/Set14", net, r, blur)
print("PSNR Set14:  ", set14_PSNR)
BSD300_PSNR = average_PSNR("../datasets/testing/BSD300", net, r, blur)
print("PSNR BSD300: ", BSD300_PSNR)
BSD500_PSNR = average_PSNR("../datasets/testing/BSD500", net, r, blur)
print("PSNR BSD500: ", BSD500_PSNR)
