import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import *
from skimage.transform import *
import os
import math
from math import floor, log10, sqrt

# hyperparameters
r = 3  # upscaling ratio
blur = 0.25  # gaussian blur (missing ???)
lr_start = 0.01  # learning rate
lr_end = 0.0001
mu = 1e-3  # threshold for lowering the lr (missing ???)
no_learning_threshold = 1e-4  # threshold for stopping training of no improvement has been made for 100 epochs
batch_size = 1
train_test_fraction = 0.8  # The part which of the data set that will be used for the training, the remainder will be used for testing (0.8 = 80%)

# parameters
dataset = "T91"
epoch_save_interval = 20
minibatch_size = 100
use_gpu = torch.cuda.is_available()

# Constants
C = 3  # colour channels
repeats = 100  # the number of consecutive epochs the improvement will have to be below the no_learning_threshold


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def PS(T, r):
    T = np.transpose(T, (1, 2, 0))
    rW = r * len(T)
    rH = r * len(T[0])
    C = len(T[0][0]) / (r * r)

    # make sure C is an integer and cast if this is the case
    assert (C == int(C))
    C = int(C)

    res = np.zeros((rW, rH, C))

    for x in range(len(res)):
        for y in range(len(res[x])):
            for c in range(len(res[x][y])):
                res[x][y][c] = \
                    T[x // r][y // r][C * r * (y % r) + C * (x % r) + c]
    return res


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


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def average_PSNR(folder, net, r):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder, filename))
        if img is not None:
            img = resize(img, ((img.shape[0] // r) * r, (img.shape[1] // r) * r))
            images.append(img)

    sumPSNR = 0
    for og_img in images:
        img = resize(og_img, (og_img.shape[0] // 3, og_img.shape[1] // 3))
        if (len(img.shape) == 2):  # convert image to rgb if it is grayscale
            img = np.stack((img, img, img), axis=2)
            og_img = np.stack((og_img, og_img, og_img), axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = torch.Tensor(img).unsqueeze(0).double()
        result = net(img).detach().numpy()
        sumPSNR += PSNR(PS(result[0], r) * 255, og_img * 255)

    return sumPSNR / len(images)


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

SUBSAMPLING_STRIDE_SIZE = 14
SUBSAMPLING_SAMPLE_SIZE = 17


# hr_dataset_path: dir to the hr_dataset png files
# downscale: downscale factor, e.g. if original image 64*64 and downscale=2 then result will be 32*32
# returns list of numpy.ndarray representing the lr_images
def lr_dataset_from_path(hr_dataset_path, downscale):
    original_filenames = os.listdir(hr_dataset_path)
    original_images = []
    for file in original_filenames:
        original_images.append(plt.imread(hr_dataset_path + '/' + file))
    return lr_images(original_images, downscale)  # ndarray of images


def torchDataloader_from_path(hr_dataset_path, downscale, gaussian_sigma):
    original_filenames = os.listdir(hr_dataset_path)
    original_images = []
    for file in original_filenames:
        original_images.append(plt.imread(hr_dataset_path + '/' + file))

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
        labeled_data.append([np.transpose(train_data[i], (2, 0, 1)), np.transpose(train_labels[i], (2, 0, 1))])
    trainDataloader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True)
    return trainDataloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(C, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, r * r * C, 3, padding=1)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = self.conv3(x)
        return x


# Start loading data
dataloader = torchDataloader_from_path('datasets/' + dataset, r, blur)
train_size = int(train_test_fraction * len(dataloader.dataset))
test_size = len(dataloader.dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataloader.dataset, [train_size, test_size])
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
print("Data loaded")

# Start training
start_time = datetime.datetime.now()
print("starting training at: " + str(start_time))

net = Net()
net.double()

if use_gpu:
    net = net.cuda()

# define loss fuction
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr_start, momentum=0)  # momentum???

losses_train = []
losses_test = []

epoch = 0
last_epoch_loss_test = float("inf")
last_epoch_loss_train = float("inf")
ni_counter = 0  # counts the amount of epochs no where no improvement has been made

now = datetime.datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
models_folder = "models"
model_name = "{}_espcnn_r{}".format(dt_string, r)

try:
    os.mkdir(models_folder + '/' + model_name)
except:
    print("Folder {} already exists, overwritting model data".format(models_folder + '/' + model_name))
model_dest = models_folder + '/' + model_name + "/model_epoch_"
best_model_dest = models_folder + '/' + model_name + "/best_model"
lr = lr_start

best_test_loss = 100000  # start with dummy value, keep track of best loss on test dataset
best_epoch = 0
while True:  # loop over the dataset multiple times
    epoch_loss_train = 0.0
    running_loss_train = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.double())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss_train += outputs.shape[0] * loss.item()
        running_loss_train += loss.item()
        if i % minibatch_size == minibatch_size - 1:  # print every 2000 mini-batches
            print('[%d, %5d] train_loss: %.5f' %
                  (epoch + 1, i + 1, running_loss_train / minibatch_size))
            running_loss_train = 0.0
    epoch_loss_train = epoch_loss_train / len(inputs)
    print(epoch + 1, epoch_loss_train)

    epoch_loss_test = 0.0
    running_loss_test = 0.0
    for i, data in enumerate(test_dataloader, 0):  # get loss on test dataset
        # get the inputs
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # forward + backward + optimize
        outputs = net(inputs.double())
        loss = criterion(outputs, labels)

        # print statistics
        epoch_loss_test += outputs.shape[0] * loss.item()
        running_loss_test += loss.item()
        if i % minibatch_size == minibatch_size - 1:  # print every 2000 mini-batches
            print('[%d, %5d] test_loss: %.5f' %
                  (epoch + 1, i + 1, running_loss_test / minibatch_size))
            running_loss_test = 0.0
    epoch_loss_test = epoch_loss_test / len(inputs)
    print(epoch + 1, epoch_loss_test)

    if epoch_loss_test < best_test_loss:  # save best model, 'best' meaning lowest loss on test set
        best_test_loss = epoch_loss_test
        torch.save(net.state_dict(), best_model_dest)  # overwrite best model so the best model filename doesn't change
        torch.save(net.state_dict(), best_model_dest + '_epoch_' + str(
            epoch + 1))  # also save with epoch number to keep history of best models
        best_epoch = epoch
        best_epoch_train_loss = epoch_loss_train

    improvement = abs(last_epoch_loss_test - epoch_loss_test)  # check for improvement with test set
    print("epoch " + str(epoch + 1) + ": improvement = " + str(improvement))
    if improvement < no_learning_threshold:
        ni_counter += 1
    else:
        ni_counter = 0

    if ni_counter >= repeats:  # stop training if no improvement has been made for 100 epochs
        break

    # If  the improvement is too small, make the learning rate smaller
    if improvement < mu and lr > lr_end:
        lr = lr / 10
        print("Learning rate decreased to:", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    losses_train.append(epoch_loss_train / len(inputs))
    losses_test.append(epoch_loss_test / len(inputs))
    last_epoch_loss_train = epoch_loss_train
    last_epoch_loss_test = epoch_loss_test

    if epoch % epoch_save_interval == 0:
        torch.save(net.state_dict(), model_dest + str(epoch + 1))
    epoch += 1

end_time = datetime.datetime.now()
print('Finished training at: ' + str(end_time))

print('Saving train and test loss')
np.save(models_folder + '/' + model_name + '/loss_train', losses_train)
np.save(models_folder + '/' + model_name + '/loss_test', losses_test)

net.cpu()
set5_PSNR = average_PSNR("datasets/testing/Set5", net, r)
set14_PSNR = average_PSNR("datasets/testing/Set14", net, r)

torch.save(net.state_dict(), "models/trained_model_" + str(set14_PSNR))

print("Finished validation \n")

print("dataset:               " + dataset)
print("psnr Set5:             " + str(set5_PSNR))
print("psnr Set14:            " + str(set14_PSNR))
print("best epoch:            " + str(best_epoch))  # epoch with the lowest loss on the test dataset
print("loss on training set:  " + str(best_epoch_train_loss))  # loss for the best epoch
print("loss on test set:      " + str(best_test_loss))  # loss for the best epoch
print("r:                     " + str(r))
print("blur:                  " + str(blur))
print("lr_start:              " + str(lr_start))
print("lr_end:                " + str(lr_end))
print("mu:                    " + str(mu))
print("no_learning_threshold: " + str(no_learning_threshold))
print("epochs:                " + str(epoch + 1))
print("training duration:     " + str(end_time - start_time))
print("batch_size:            " + str(batch_size))
print("train_test_fraction:   " + str(train_test_fraction))
print("model saved as:        " + "trained_model_" + str(set14_PSNR))
