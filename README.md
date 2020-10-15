# Honours Project - Ben Maxwell
## Project Overview
An analysis of cracks in concrete structures using a neural network with an investigation into activation functions.
Further research looks into instance segmentation.

## Network

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

![img/u-net-architecture.png](img/u-net-architecture.png)

The original generic implementation of this network is from: https://github.com/zhixuhao/unet

Acreditation has been given in the report for both the original implementation and the dataset.

## Data

The dataset that will be used to test the neural network will be CrackForest, located at: https://github.com/cuilimeng/CrackForest-dataset.

You can find it in folder data/crackForest.

The data will be trained using these 480*320 images, eventually I might convert them to a more standard resolution


## Training

The model is trained for 5 epochs.

After 5 epochs, calculated accuracy is about 0.97.

Loss function for the training is basically just a binary crossentropy.

Later in the development of the project, a function will control when to stop the network training after a number of epochs without progress allowing for peak efficiency

# How to use

## Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras >= 1.0 (GPU Enabled version)
* Python = 3.5
* Scikit-image (Scimage)
* Anaconda (Python Environment)
* Install all dependencies through anaconda and select the conda environment as a Python Interpreter

### Installing Anaconda Environment
* Install Anaconda: https://www.anaconda.com/products/individual
* Open Anaconda Prompt with 'Run as Adminstrator'
* Type 'conda create -n KerasGPU python=3.5' and follow install instructions
* Type 'conda activate KerasGPU'
* Type 'conda install tensorflow' (This is a cpu only version)
* Type 'conda install keras-gpu'
* Type 'conda install scikit-image'


### Run main.py

You will see the predicted results of test image in data/membrane/test


## Results

Use the trained model to do segmentation on test images, the result is statisfactory.

![img/0test.png](img/0test.png)

![img/0label.png](img/0label.png)


## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
