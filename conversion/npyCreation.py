import numpy as np 
import os
import glob
import cv2
import skimage.io as io
import skimage.transform as trans

def adjustData(img,mask,flag_multi_class,num_class, target_size = (320,480)):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #index = np.where(mask == i)
            index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            new_mask[index_mask] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        #img = normalizeData(img) As the new way of normalizing data handles this it is no longer
        img = claheEqualization(img)
        img = adjustGamma(img, 1.2)
        img = trans.resize(img, target_size)
        mask = trans.resize(mask, target_size)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        #img = img / 255
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        #mask = mask / 255
        mask[mask > 0.32] = 1
        mask[mask <= 0.32] = 0
    return (img,mask)

def normalizeData(img):
    assert (len(img.shape)==4) 
    assert (img.shape[1]==1) 
    imgNormalized = np.empty(img.shape)
    imgStd = np.std(img)
    imgMean = np.mean(img)
    imgNormalized = (img-imgMean)/imgStd
    for i in range(img.shape[0]):
        imgNormalized[i] = ((imgNormalized[i] - np.min(imgNormalized[i])) / (np.max(imgNormalized[i])-np.min(imgNormalized[i])))*255 # Essentially the same as what's happening in adjust data
    return imgNormalized # I wasn't sure that my normalisation method was fully functional so I added this from a model that I knew worked for reliability

def claheEqualization(img):
    assert (len(img.shape)==4)
    assert (img.shape[1]==1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgEqualized = np.empty(img.shape)
    for i in range(img.shape[0]):
        imgEqualized[i,0] = clahe.apply(np.array(img[i,0], dtype = np.uint8))
    return imgEqualized

def adjustGamma(img, gamma):
    assert (len(img.shape)==4)
    assert (img.shape[1]==1)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    newImg = np.empty(img.shape)
    for i in range(img.shape[0]):
        newImg[i,0] = cv2.LUT(np.array(img[i,0], dtype = np.uint8), table)
    return newImg

image_path = ("B:/Downloads/Honours/data/crackForest/train/image")
mask_path = ("B:/Downloads/Honours/data/crackForest/train/label")
#image_path = ("B:/Downloads/Honours/data/crackconcrete/train/image")
#mask_path = ("B:/Downloads/Honours/data/crackconcrete/train/label")
image_prefix = ""
mask_prefix = ""
image_as_gray = True
mask_as_gray = True
flag_multi_class = False
num_class = 2
#target_size = (160,240)

image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
image_arr = []
mask_arr = []

for index,item in enumerate(image_name_arr):
    img = io.imread(item,as_gray = image_as_gray)
    img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
    mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
    mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
    #img = trans.resize(img,target_size)
    img,mask = adjustData(img,mask,flag_multi_class,num_class)
    image_arr.append(img)
    mask_arr.append(mask)
image_arr = np.array(image_arr)
mask_arr = np.array(mask_arr)
np.save("data/crackForest/npydata/image_training.npy",image_arr)
np.save("data/crackForest/npydata/mask_training.npy",mask_arr)
print("Success")