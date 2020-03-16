# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:20:18 2020

@author: olivi
"""

# -*- coding: utf-8 -*-
"""
see TEST CODE at the bottom for an example of how to use
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

#blur, downscale, subsample
def originalImagesToTrainingData(original_images, downscale_factor, gaussian_sigma):
    train_data = []    
    for img in  range(len(original_images)):        
        img_blurred = gaussian(original_images[img], sigma = gaussian_sigma, multichannel=True)#multichannel blurr so that 3rd channel is not blurred
        img_downscaled = resize(img_blurred, (img_blurred.shape[0]//downscale_factor,img_blurred.shape[1]//downscale_factor))        
        temp_subsamples = subsample(img_downscaled, 1) #set downscale factor to 1 because image has already been downscaled
        for sub in temp_subsamples:
            train_data.append(sub)
    return train_data

#subsample, reverse shuffle
def originalImagesToTrainingLabels(original_images, downscale_factor):
    train_labels = []
    for i in range(len(original_images)):        
        temp_subsamples = subsample(original_images[i], downscale_factor)
        for sub in temp_subsamples:
            train_labels.append(PS_inv(sub, downscale_factor))
    return train_labels


#creates train_x and train_y from the images in the dataset path folder using the downscale_factor, saves training data in folder save_to
def createAndSaveTrainDataset(original_dataset_path, downscale_factor, gaussian_sigma, save_to='train_data'):    
    original_images = getImagesFromPath(original_dataset_path)
    train_x = originalImagesToTrainingData(original_images, downscale_factor, gaussian_sigma)
    train_y = originalImagesToTrainingLabels(original_images, downscale_factor)
    try:
        os.mkdir(save_to)
    except FileExistsError:
        print("> WARNING Directory: /" + save_to + " already exists.\n>> Overwriting data...")
    else:
        print("... Creating directory: " + save_to)
    np.save(save_to + '/train_x', train_x)
    np.save(save_to + '/train_y', train_y)
    print('Successfully saved train_x.npy and train_y.npy to ' + save_to + '/')
    
#assume train data file names are train_x.npy and train_y.npy, return pytorch Dataloader
def storedTrainDatasetToTorchDataloader(train_dataset_path):
    train_x = np.load(train_dataset_path + '/train_x.npy')
    train_y = np.load(train_dataset_path + '/train_y.npy')
    return toDataloader(train_x, train_y)
        
        
def getImagesFromPath(images_path):
    original_filenames= os.listdir(images_path)
    original_images = []
    for file in original_filenames:
        original_images.append(image.imread(images_path + '/' + file))
    return original_images
    
    


def torchDataloader_from_path(hr_dataset_path, downscale, gaussian_sigma):
    original_filenames= os.listdir(hr_dataset_path)
    original_images = []
    for file in original_filenames:
        original_images.append(image.imread(hr_dataset_path + '/' + file))
        
    #subsample
    subsamples_hr = []
    subsamples_hr_rev_shuff = []
    for i in range(len(original_images)):   
        temp_subsamples = subsample(original_images[i], downscale)                 
        subsamples_hr += temp_subsamples
        for sample_indx in range(len(temp_subsamples)):
            subsamples_hr_rev_shuff.append(PS_inv(temp_subsamples[sample_indx], downscale))  #labels
    lr_dataset = lr_images(subsamples_hr, downscale, gaussian_sigma) #ndarray of images

    return toDataloader(lr_dataset, subsamples_hr_rev_shuff)




#extract a 17r*17r subsample from original image, no overlap so every pixel appears at most once in output
def subsample(image_real, downscale):
    subsample_size = SUBSAMPLING_SAMPLE_SIZE * downscale   
    subsample_stride = SUBSAMPLING_STRIDE_SIZE * downscale   
    subsamples = []
    for y in range(math.floor((image_real.shape[0] - (subsample_size - subsample_stride)) / subsample_stride)):
        for x in range(math.floor((image_real.shape[1] - (subsample_size - subsample_stride)) / subsample_stride)):
            ss = image_real[(y*subsample_stride):(y*subsample_stride)+subsample_size, (x*subsample_stride):(x*subsample_stride)+subsample_size]
            subsamples.append(ss)
            
    return subsamples

#returns a torch Dataloader (to iterate over training data) using the training data samples and traing data labels
def toDataloader(train_data, train_labels):
    labeled_data = []  
    for i in range(len(train_data)): 
        labeled_data.append([train_data[i], train_labels[i]])            
    trainDataloader = DataLoader(labeled_data, batch_size=3)
    return trainDataloader

def PS_inv(img, r):
    r2 = r*r
    W = len(img)/r
    H = len(img[0])/r
    C = len(img[0][0])
    Cr2 = C*r2

    # Make sure H and W are integers
    assert(int(H) == H and int(W) == W)
    H, W = int(H), int(W)

    res = np.zeros((W, H, Cr2))

    for x in range(len(img)):
        for y in range(len(img[x])):
            for c in range(len(img[x][y])):
                res[x // r][y // r][C*r*(y % r) + C*(x % r) + c] = img[x][y][c]
    return res

    
#---TEST CODE---
if __name__ == "__main__":   
    print('***Running test***\n')
    print('creating train_data dataset')
    dataset_path = '../datasets/test'
    train_data_path = 'train_data'
    createAndSaveTrainDataset(dataset_path, downscale_factor=3, gaussian_sigma=3, save_to=train_data_path)
    dataloader = storedTrainDatasetToTorchDataloader(train_data_path)
    i1, l1 = next(iter(dataloader))
    print(i1.shape)
    print(l1.shape)
    print('\n***Finished***')
