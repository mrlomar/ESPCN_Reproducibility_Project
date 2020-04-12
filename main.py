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
from src.dataloader import torchDataloader_from_path
from src.metrics import *
from src.espcn import *
import csv

# hyperparameters
r = 3  # upscaling ratio
blur = 0.1  # gaussian blur (missing ???)
lr_start = 0.01  # learning rate
lr_end = 0.0001
mu = 1e-5  # threshold for lowering the lr (missing ???)
no_learning_threshold = 1e-7  # threshold for stopping training of no improvement has been made for 100 epochs
batch_size = 1
train_test_fraction = 0.8  # The part which of the data set that will be used for the training, the remainder will be used for testing (0.8 = 80%)

# parameters
dataset = "T91"
epoch_save_interval = 100
minibatch_size = 100
use_gpu = torch.cuda.is_available()

# Constants
C = 3  # colour channels
repeats = 100  # the number of consecutive epochs the improvement will have to be below the no_learning_threshold


# Start loading data
dataloader = torchDataloader_from_path('./datasets/' + dataset, r, blur, batch_size)
train_size = int(train_test_fraction * len(dataloader.dataset))
test_size = len(dataloader.dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataloader.dataset, [train_size, test_size])
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
print("Data loaded")

# Start training
start_time = datetime.datetime.now()
print("starting training at: " + str(start_time))

net = Net(r, C)
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
try:
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
	    epoch_loss_train = epoch_loss_train / len(train_dataloader.dataset)
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
	    epoch_loss_test = epoch_loss_test / len(test_dataloader.dataset)
	    print(epoch + 1, epoch_loss_test)

	    improvement = best_test_loss - epoch_loss_test

	    if epoch_loss_test < best_test_loss:  # save best model, 'best' meaning lowest loss on test set
		best_test_loss = epoch_loss_test
		torch.save(net.state_dict(), best_model_dest)  # overwrite best model so the best model filename doesn't change
		best_epoch = epoch
		best_epoch_train_loss = epoch_loss_train

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

	    losses_train.append(epoch_loss_train)
	    losses_test.append(epoch_loss_test)
	    last_epoch_loss_train = epoch_loss_train
	    last_epoch_loss_test = epoch_loss_test

	    if epoch % epoch_save_interval == 0:
		torch.save(net.state_dict(), model_dest + str(epoch + 1))
	    epoch += 1
except KeyboardInterrupt:
    print("Press Ctrl-C to terminate while statement")
    pass

    print('Saving train and test loss')
    np.save(models_folder + '/' + model_name + '/loss_train', losses_train)
    np.save(models_folder + '/' + model_name + '/loss_test', losses_test)

end_time = datetime.datetime.now()
print('Finished training at: ' + str(end_time))

net.load_state_dict(torch.load(best_model_dest))
net.eval()

net.cpu()
set5_PSNR = average_PSNR("./datasets/testing/Set5", net, r, blur)
set14_PSNR = average_PSNR("./datasets/testing/Set14", net, r, blur)

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
print("model:                 " + model_name)

with open(models_folder + '/' + model_name + '/results.csv', mode='w') as csv_file:
	fieldnames = ['dataset', 'psnr_Set5', 'psnr_Set14', 'best_epoch', 'training_loss', 'test_loss', 'r', 'blur', 'lr_start', 'lr_end', 'mu', 'no_learning_threshold', 'epochs', 'training_duration', 'batch_size', 'train_test_fraction', 'model']
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	writer.writeheader()
	writer.writerow({
		'dataset': dataset,
		'psnr_Set5': set5_PSNR,
		'psnr_Set14': set14_PSNR,
		'best_epoch': best_epoch,
		'training_loss': best_epoch_train_loss,
		'test_loss': best_test_loss,
		'r': r,
		'blur': blur,
		'lr_start': lr_start,
		'lr_end': lr_end,
		'mu': mu,
		'no_learning_threshold': no_learning_threshold,
		'epochs': (epoch + 1),
		'training_duration': (end_time - start_time),
		'batch_size': batch_size,
		'train_test_fraction': train_test_fraction,
		'model': model_name})
