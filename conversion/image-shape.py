import cv2
import numpy as np
# This was created to check the image shape of test data quickly
# This allowed me to identify issues with datasets and helped to make accurate predictions

#img = cv2.imread('B:/Downloads/Honours/data/crackForest/train/label/001.png', cv2.IMREAD_UNCHANGED)
img = cv2.imread('B:/Downloads/Honours/data/crackconcrete/train/image/001.png', cv2.IMREAD_UNCHANGED)
#img = cv2.imread('B:/Downloads/NewLabel/001.png', cv2.IMREAD_UNCHANGED)

if len(img.shape) < 3:
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    channels = 'Greyscale'
if len(img.shape)>= 3:
    dimensions = img.shape
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

print('Image Dimensions: ',dimensions)
print('Image Height: ', height)
print('Image Width: ', width)
print('Channels: ',channels)