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

#hr_dataset_path: dir to the hr_dataset png files
#downscale: downscale factor, e.g. if original image 64*64 and downscale=2 then result will be 32*32
#returns list of numpy.ndarray representing the lr_images
def lr_dataset_from_path(hr_dataset_path, downscale):
    original_filenames= os.listdir(hr_dataset_path)
    original_images = []
    for file in original_filenames:
        original_images.append(image.imread(hr_dataset_path + '/' + file))
    return lr_images(original_images, downscale) #ndarray of images


# Takes list of images and provide LR images in form of numpy array
def lr_images(images_real , downscale, gaussianSigma):
    
    lr_images = []
    for img in  range(len(images_real)): 
        img_blurred = skimage.filters.gaussian(images_real[img], sigma = gaussianSigma, multichannel=True)#multichannel blurr so that 3rd channel is not blurred
        lr_images.append(skimage.transform.resize(img_blurred, (img_blurred.shape[0]//downscale,img_blurred.shape[1]//downscale)))
    return lr_images



# ---DEBUG--- uncomment to show first image
#pyplot.imshow(original_images[0])
#pyplot.imshow(lr_dataset[0])


# ---SAVE LR FILES AS PNG--- 
save_png = True #save lr png images to folder: param_dir_lr_png

#save images to file
if(save_png):   
    print("save_png = True")
    # ---SET PARAMS---
    param_gaussianSigma = 3
    param_downscale = 3 #downscale factor
    param_dir_hr_png = '../datasets/T91' #dir of originl dataset
    param_dir_lr_png = 'dataset_lr' #dir of output dataset
    
    print("... Reading images in: " + param_dir_hr_png)
    original_filenames= os.listdir(param_dir_hr_png)
    original_images = []
    for file in original_filenames:
        original_images.append(image.imread(param_dir_hr_png + '/' + file))
    
    print("... Creating lr dataset")
    lr_dataset = lr_images(original_images, param_downscale, param_gaussianSigma) #ndarray of images

    try:
        os.mkdir(param_dir_lr_png)
    except FileExistsError:
        print("> WARNING Directory: /" + param_dir_lr_png + " already exists.\n>> Overwriting data...")
    else:
        print("... Creating directory: " + param_dir_lr_png)
    print("... Saving lr images...")
    for i in range(len(lr_dataset)):
        image.imsave(param_dir_lr_png + '/' + 'lr_' + original_filenames[i], lr_dataset[i])
    print("... lr images saved to: " + '/' +  param_dir_lr_png)
    
print("Finished")
