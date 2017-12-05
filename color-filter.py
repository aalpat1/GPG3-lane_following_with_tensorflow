
# coding: utf-8

# In[13]:

import cv2
import numpy as np


# In[21]:

def red_filtered(image):
    lbound = np.array([0, 50, 50], dtype = np.uint8)
    ubound = np.array([10, 255, 255], dtype = np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lbound, ubound)
    rgb_img = np.zeros((mask.shape[0], mask.shape[1], 3), 'uint8')
    rgb_img[..., 0] = mask
    rgb_img[..., 1] = mask
    rgb_img[..., 2] = mask
    return mask, rgb_array
    


# In[24]:

def filter_images:
    # Read images from directory here
    img = cv2.imread("sample.jpg")
    #
    
    result = red_filtered(img)
    bit_mask = result[0] # Will be of shape n * m * 1, where n * m is dimension of image, since only 1 binary channel
    rgb_array = result[1] # Will be of shape n * m * 3, because bit_mask will be converted to a black and white image


# In[ ]:



