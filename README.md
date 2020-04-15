# ESPCN_Reproducibility_Project
This repository consists of our attempt to reproduce the paper [Real-Time Single Image and Video Super-Resolution Using an EfficientSub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158v2.pdf) for the course CS4240 Deep Learning at Delft University of Technology.  
The code used for training during the project are the *.py* files and a blogpost is available that gives more insight into our approach.  

---

## Training the network

To train the network one only has to run *main.py* (for example with `python main.py`).  
In order to use other parameters the hyperparameters withing *main.py* (as of writing line 18-35) can be edited before running the program.  
  
During training intermediate results can be inspected using *show.py*, where a folder containing the network data can be passed as a system argument.  
An example of using *show.py* is `python show.py '2020-04-08_18-18-51_espcnn_r3'`.  

The best model, intermediate models (every 100 epochs) and the training/test loss are saved during training in a folder named 'models', to run locally this folder needs to be created.

---

## The Data
The data sets used for training and evaluation are publically available.  
set5, set14 http://vllab.ucmerced.edu/wlai24/LapSRN/  
bsd500 https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html  
bsd300 https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/  

To evaluate our model these datasets need to be downloaded and the directories in the main.py need to be updated to match the corresponding dataset folders.

---

## Results
We were unable to fully reproduce the results from [the paper](https://arxiv.org/pdf/1609.05158v2.pdf).
However, we think that using our code the results from the paper could be reproducible if the hyperparameters are set correctly and enough training time is given.

---

## Blog
available [here](https://github.com/mrlomar/ESPCN_Reproducibility_Project/blob/master/blogpost.ipynb).
