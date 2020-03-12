# -*- coding: utf-8 -*-
"""
Downsample images
- gaussian blur
- resize by downsample factor (using interpolation)

How To Use:
    function lr_dataset_from_path takes a path to the dataset of HR image png files and returns an ndarray to use for training the model
    
For debugging/showing examples:
    (see bottom of file)
    save_png set to True to save resulting lr images in specified directory.
    !check the param_ varaiables
"""

import numpy as np
from scipy import misc
from PIL import Image, ImageFilter
from matplotlib import image
from matplotlib import pyplot
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import *
from skimage.transform import *
import os
import math
from torch.utils.data import DataLoader
from math import floor

SUBSAMPLING_STRIDE_SIZE = 14
SUBSAMPLING_SAMPLE_SIZE = 17


# hr_dataset_path: dir to the hr_dataset png files
# downscale: downscale factor, e.g. if original image 64*64 and downscale=2 then result will be 32*32
# returns list of numpy.ndarray representing the lr_images
def lr_dataset_from_path(hr_dataset_path, downscale):
    original_filenames = os.listdir(hr_dataset_path)
    original_images = []
    for file in original_filenames:
        original_images.append(image.imread(hr_dataset_path + '/' + file))
    return lr_images(original_images, downscale)  # ndarray of images


def torchDataloader_from_path(hr_dataset_path, downscale, gaussian_sigma):
    original_filenames = os.listdir(hr_dataset_path)
    original_images = []
    for file in original_filenames:
        original_images.append(image.imread(hr_dataset_path + '/' + file))

    # subsample
    subsamples_hr = []
    subsamples_hr_rev_shuff = []
    for i in range(len(original_images)):
        temp_subsamples = subsample(original_images[i], downscale)
        subsamples_hr += temp_subsamples
        for sample_indx in range(len(temp_subsamples)):
            subsamples_hr_rev_shuff.append(PS_inv(temp_subsamples[sample_indx], downscale))  # labels
    lr_dataset = lr_images(subsamples_hr, downscale, gaussian_sigma)  # ndarray of images

    return toDataloader(lr_dataset, subsamples_hr_rev_shuff)


# Takes list of images and provide LR images in form of numpy array
def lr_images(images_real, downscale, gaussianSigma):
    lr_images = []
    for img in range(len(images_real)):
        img_blurred = gaussian(images_real[img], sigma=gaussianSigma,
                               multichannel=True)  # multichannel blurr so that 3rd channel is not blurred
        lr_images.append(resize(img_blurred, (img_blurred.shape[0] // downscale, img_blurred.shape[1] // downscale)))
    return lr_images


# extract a 17r*17r subsample from original image, no overlap so every pixel appears at most once in output
def subsample(image_real, downscale):
    subsample_size = SUBSAMPLING_SAMPLE_SIZE * downscale
    subsample_stride = SUBSAMPLING_STRIDE_SIZE * downscale
    subsamples = []
    for y in range(math.floor((image_real.shape[0] - (subsample_size - subsample_stride)) / subsample_stride)):
        for x in range(math.floor((image_real.shape[1] - (subsample_size - subsample_stride)) / subsample_stride)):
            ss = image_real[(y * subsample_stride):(y * subsample_stride) + subsample_size,
                 (x * subsample_stride):(x * subsample_stride) + subsample_size]
            subsamples.append(ss)

    return subsamples


# returns a torch Dataloader (to iterate over training data) using the training data samples and traing data labels
def toDataloader(train_data, train_labels):
    labeled_data = []
    for i in range(len(train_data)):
        labeled_data.append([train_data[i], train_labels[i]])
    trainDataloader = DataLoader(labeled_data, batch_size=3)
    return trainDataloader


def PS_inv(img, r):
    r2 = r * r
    W = len(img) / r
    H = len(img[0]) / r
    C = len(img[0][0])
    Cr2 = C * r2

    # Make sure H and W are integers
    assert (int(H) == H and int(W) == W)
    H, W = int(H), int(W)

    res = np.zeros((W, H, Cr2))

    for x in range(len(img)):
        for y in range(len(img[x])):
            for c in range(len(img[x][y])):
                res[x // r][y // r][C * r * (y % r) + C * (x % r) + c] = img[x][y][c]
    return res


# ---DEBUG--- uncomment to show first image
# pyplot.imshow(original_images[0])
# pyplot.imshow(lr_dataset[0])

# ----TEST DATALOADER ----
dl = torchDataloader_from_path('../datasets/test', 3, 5)

print("TESTIng datalaoder iter")
i1, l1 = next(iter(dl))
print(i1.shape)
print(l1.shape)

print("DONE!!!")

# ---SAVE LR FILES AS PNG---
save_png = False  # save lr png images to folder: param_dir_lr_png

# save images to file
if (save_png):
    print("save_png = True")
    # ---SET PARAMS---
    param_gaussianSigma = 3
    param_downscale = 3  # downscale factor
    param_dir_hr_png = '../datasets/T91'  # dir of originl dataset
    param_dir_lr_png = 'dataset_lr'  # dir of output dataset

    print("... Reading images in: " + param_dir_hr_png)
    original_filenames = os.listdir(param_dir_hr_png)
    original_images = []
    for file in original_filenames:
        read_img = image.imread(param_dir_hr_png + '/' + file)
        original_images.append(read_img)

    # subsample
    subsamples_hr = []
    subsample_filenames = []
    subsamples_hr_rev_shuff = []
    for i in range(len(original_images)):
        temp_subsamples = subsample(original_images[i], param_downscale)
        subsamples_hr += temp_subsamples
        for sample_indx in range(len(temp_subsamples)):
            subsamples_hr_rev_shuff.append(PS_inv(temp_subsamples[sample_indx], param_downscale))
            subsample_filenames.append(original_filenames[i][:-4] + "_" + str(sample_indx) + '.png')

    print("... Creating lr dataset")
    lr_dataset = lr_images(subsamples_hr, param_downscale, param_gaussianSigma)  # ndarray of images

    try:
        os.mkdir(param_dir_lr_png)
    except FileExistsError:
        print("> WARNING Directory: /" + param_dir_lr_png + " already exists.\n>> Overwriting data...")
    else:
        print("... Creating directory: " + param_dir_lr_png)

    try:
        os.mkdir(param_dir_lr_png + "/subsamples_lr")
    except FileExistsError:
        print(
            "> WARNING Directory: /" + param_dir_lr_png + "/subsamples_lr" + " already exists.\n>> Overwriting data...")

    try:
        os.mkdir(param_dir_lr_png + "/subsamples_hr")
    except FileExistsError:
        print(
            "> WARNING Directory: /" + param_dir_lr_png + "/subsamples_hr" + " already exists.\n>> Overwriting data...")

    try:
        os.mkdir(param_dir_lr_png + "/subsamples_hr_rev_shuff")
    except FileExistsError:
        print(
            "> WARNING Directory: /" + param_dir_lr_png + "/subsamples_hr_rev_shuff" + " already exists.\n>> Overwriting data...")

    print("... Saving subsampled lr images (training samples)...")
    for i in range(len(lr_dataset)):
        image.imsave(param_dir_lr_png + "/subsamples_lr" + '/' + 'lr_ss_' + subsample_filenames[i], lr_dataset[i])

    print("... Saving subsampled hr images (training ground truth labels)...")
    for i in range(len(subsamples_hr)):
        image.imsave(param_dir_lr_png + "/subsamples_hr" + '/' + 'hr_ss_' + subsample_filenames[i], subsamples_hr[i])
    print("... images saved to: " + '/' + param_dir_lr_png)

    print("")
    for i in range(len(subsamples_hr_rev_shuff)):
        np.save(param_dir_lr_png + "/subsamples_hr_rev_shuff" + '/' + 'hr_ss_' + subsample_filenames[i],
                subsamples_hr_rev_shuff[i])

print("Finished")
