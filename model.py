import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras import backend as keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from custom_activations import swish

def unet(pretrained_weights = None,input_size = (320,480,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


# Convolutional neural network created based upon the U-Net architecture
def unet_cracks(pretrained_weights = None,input_size = (320,480,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64,3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64,3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128,3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128,3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256,3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256,3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512,3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512,3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(pool4))
    merge7 = Concatenate(axis=3)([conv3,up7])
    conv7 = Conv2D(256,3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256,3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis=3)([conv2,up8])
    conv8 = Conv2D(128,3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128,3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis=3)([conv1,up9])
    conv9 = Conv2D(64,3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64,3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2,3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9) 

    conv10 = Conv2D(1,1, activation='sigmoid')(conv9) # Returns the prediction as either 1 or 0

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

# Smaller network still based off of U-Net architecture
def simple_unet(pretrained_weights = None,input_size = (320,480,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64,3, activation='swish', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64,3, activation='swish', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128,3, activation='swish', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128,3, activation='swish', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256,3, activation='swish', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256,3, activation='swish', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512,3, activation='swish', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512,3, activation='swish', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(256, 2, activation='swish', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(drop4))
    trans5 = Conv2DTranspose(256, (2,2), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(up5)
    merge5 = Concatenate(axis=3)([conv3,trans5])
    conv5 = Conv2D(256,3, activation='swish', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(256,3, activation='swish', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(128, 2, activation='swish', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv5))
    trans6 = Conv2DTranspose(128, (2,2), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(up6)
    merge6 = Concatenate(axis=3)([conv2,trans6])
    conv6 = Conv2D(128,3, activation='swish', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(128,3, activation='swish', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, 2, activation='swish', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv6))
    trans7 = Conv2DTranspose(64, (2,2), strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(up7)
    merge7 = Concatenate(axis=3)([conv1,trans7])
    conv7 = Conv2D(64,3, activation='swish', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64,3, activation='swish', padding='same', kernel_initializer='he_normal')(conv7)

    conv7 = Conv2D(2,3, activation='swish', padding='same', kernel_initializer='he_normal')(conv7) 
    conv8 = Conv2D(1,1, activation='sigmoid')(conv7) # Returns the prediction as either 1 or 0

    model = Model(input = inputs, output = conv8)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def crf_unet(pretrained_weights = None,input_size = (320,480,1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32,3, activation='swish', padding='same', kernel_initializer='he_normal')(inputs)
    drop1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32,3, activation='swish', padding='same', kernel_initializer='he_normal')(drop1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64,3, activation='swish', padding='same', kernel_initializer='he_normal')(pool1)
    drop2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64,3, activation='swish', padding='same', kernel_initializer='he_normal')(drop2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128,3, activation='swish', padding='same', kernel_initializer='he_normal')(pool2)
    drop3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128,3, activation='swish', padding='same', kernel_initializer='he_normal')(drop3)

    up3 = UpSampling2D(size=(2, 2))(conv3)
    up3 = Concatenate(axis=3)([conv2,up3])

    conv4 = Conv2D(64,3, activation='swish', padding='same', kernel_initializer='he_normal')(up3)
    drop4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64,3, activation='swish', padding='same', kernel_initializer='he_normal')(drop4)

    up5 = UpSampling2D(size = (2,2))(conv4)
    up5 = Concatenate(axis=3)([conv1,up5])

    conv5 = Conv2D(32,3, activation='swish', padding='same', kernel_initializer='he_normal')(up5)
    drop5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32,3, activation='swish', padding='same', kernel_initializer='he_normal')(drop5)

    conv6 = Conv2D(2, 1, activation='swish', padding='same', kernel_initializer='he_normal')(conv5)

    conv7 = Conv2D(1,1, activation='sigmoid')(conv6)

    model = Model(input = inputs, output = conv7)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

# Smaller network still based off of U-Net architecture using LeakyReLu
def lrcrf_unet(pretrained_weights = None,input_size = (320,480,1)):
    inputs = Input(input_size)
    leaky_relu = LeakyReLU(alpha=0.2)

    conv1 = Conv2D(32,3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = leaky_relu(conv1)
    drop1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32,3, padding='same', kernel_initializer='he_normal')(drop1)
    conv1 = leaky_relu(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64,3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = leaky_relu(conv2)
    drop2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64,3, padding='same', kernel_initializer='he_normal')(drop2)
    conv2 = leaky_relu(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128,3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = leaky_relu(conv3)
    drop3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128,3, padding='same', kernel_initializer='he_normal')(drop3)
    conv3 = leaky_relu(conv3)

    up3 = UpSampling2D(size=(2, 2))(conv3)
    up3 = Concatenate(axis=3)([conv2,up3])

    conv4 = Conv2D(64,3, padding='same', kernel_initializer='he_normal')(up3)
    conv4 = leaky_relu(conv4)
    drop4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64,3, padding='same', kernel_initializer='he_normal')(drop4)
    conv4 = leaky_relu(conv4)

    up5 = UpSampling2D(size = (2,2))(conv4)
    up5 = Concatenate(axis=3)([conv1,up5])

    conv5 = Conv2D(32,3, padding='same', kernel_initializer='he_normal')(up5)
    conv5 = leaky_relu(conv5)
    drop5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32,3, padding='same', kernel_initializer='he_normal')(drop5)
    conv5 = leaky_relu(conv5)

    conv6 = Conv2D(2, 1, padding='same', kernel_initializer='he_normal')(conv5)
    conv6 = leaky_relu(conv6)

    conv7 = Conv2D(1,1, activation='sigmoid')(conv6)

    model = Model(input = inputs, output = conv7)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model