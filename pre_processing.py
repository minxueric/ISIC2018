###################################################
#
#   Script to pre-process the original imgs
#
##################################################


import numpy as np
from PIL import Image
import cv2

from help_functions import *


#My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #my preprocessing:
    #train_imgs = histo_equalized(data)
    #train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(data)
    #train_imgs = adjust_gamma(data, 1.2)
    #train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==3)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    imgs = imgs.transpose(0,2,3,1)
    for i in range(imgs.shape[0]):
        img_yuv = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2YUV)
        # equalization on HSV channel V
        img_yuv[:,:,2] = cv2.equalizeHist(img_yuv[:,:,2].astype(np.uint8))
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        img_output = img_output.transpose(2,0,1)
        imgs_equalized[i] = img_output
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==3)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    imgs = imgs.transpose(0,2,3,1)
    for i in range(imgs.shape[0]):
        img_yuv = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2YUV)
        img_yuv[:,:,2] = clahe.apply(img_yuv[:,:,2].astype(np.uint8))
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        img_output = img_output.transpose(2,0,1)
        imgs_equalized[i] = img_output
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

if __name__ == "__main__":
    img = cv2.imread('./datasets/ISIC2018_Task1-2_Training_Input/ISIC_0000007.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img.transpose((2,0,1)).astype(np.float32)
    print img.shape

    plt.figure()
    plt.imshow(img.transpose(1,2,0).astype(np.uint8))
    plt.show()

    img = my_PreProc(img[np.newaxis, :,:,:])[0]
    print img.shape

    plt.figure()
    plt.imshow(img.transpose(1,2,0).astype(np.uint8))
    plt.show()
