import scipy.io
import os
import glob
import cv2
import numpy as np

# As there was numerous issues with Matlab, I resorted to using this script
# Once I get some time I will automate it to load in each mat file
# Then use a for loop to loop through each file and write them to the folder
num_images = 117

for i in range (0,num_images+1):
    mat = scipy.io.loadmat('B:/Downloads/CrackForest-dataset/gtNames/{}.mat'.format(i))

    np_seg = mat['groundTruth'][0][0][0]
    (y, x) = np.where(np_seg == 2)
    np_seg[y, x] = 255 # Perfect division would = 1 therefore equaling the crack
    (y, x) = np.where(np_seg == 1)
    np_seg[y, x] = 0 # Dividing by 255 would return 0 therefore giving us the concrete
    (y, x) = np.where(np_seg == 3)
    np_seg[y, x] = 0
    (y,x) = np.where(np_seg == 4) # Issue with certain images showing a segment inbetween webbed cracks
    np_seg[y, x] = 0 # Not looking for that data so it's removed

    cv2.imwrite('B:/Downloads/NewLabel/{}.png'.format(i), np_seg)
print("Success")