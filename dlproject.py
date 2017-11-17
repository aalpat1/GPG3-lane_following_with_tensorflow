
# coding: utf-8

# In[2]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import glob
import sys
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import os


# In[3]:


def get_img_array(path):
    """
    Given path of image, returns it's numpy array
    """
    return scipy.misc.imread(path)

def check_files(folder):
    """
    Given path to folder, returns list of files in it
    """
    filenames = [file for file in glob.glob(folder+'*/*')]
    if(len(filenames)==0):
        return False
    return True

def get_files(folder):
    """
    Given path to folder, returns list of files in it
    """
    filenames = [file for file in glob.glob(folder+'*/*')]
    filenames.sort()
    return filenames

def get_label(label2id, label):
    """
    Returns label for a folder
    """
    if label in label2id:
        return label2id[label]
    else:
        sys.exit("Invalid label: " + label)


# In[4]:


# Functions to load data, DO NOT change these

def get_labels(folder, label2id):
    """
    Returns vector of labels extracted from filenames of all files in folder
    :param folder: path to data folder
    :param label2id: mapping of text labels to numeric ids. (Eg: automobile -> 0)
    """
    files = get_files(folder)
    y = []
    for f in files:
        y.append(get_label(f,label2id))
    return np.array(y)

def one_hot(y, num_classes=10):
    """
    Converts each label index in y to vector with one_hot encoding
    """
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[y] = 1
    return y_one_hot.T

def get_label_mapping(label_file):
    """
    Returns mappings of label to index and index to label
    The input file has list of labels, each on a separate line.
    """
    with open(label_file, 'r') as f:
        id2label = f.readlines()
        id2label = [l.strip() for l in id2label]
    label2id = {}
    count = 0
    for label in id2label:
        label2id[label] = count
        count += 1
    return id2label, label2id

def get_images(folder):
    """
    returns numpy array of all samples in folder
    each column is a sample resized to 30x30 and flattened
    """
    files = get_files(folder)
    images = []
    count = 0
    
    for f in files:
        count += 1
        if count % 10000 == 0:
            print("Loaded {}/{}".format(count,len(files)))
        img_arr = get_img_array(f)
        img_arr = img_arr.flatten() / 255.0
        images.append(img_arr)
    X = np.column_stack(images)

    return X

def get_train_data(data_root_path, label_mapping_path):
    """
    Return X and y
    """
    X = []
    y = []
    labels = os.listdir(data_root_path)
    id2label, label2id = get_label_mapping(label_mapping_path)
    print(label2id)
    for label in labels:
        train_data_path = data_root_path + label
        if(check_files(train_data_path)) :
            X_temp = get_images(train_data_path)
            size = X_temp[0].size
            y.extend(np.full((size), get_label(label2id, label)))
            if(len(X)==0):
                X = X_temp
            else:
                X = np.concatenate((X,X_temp), axis = 1)

    return X, np.array(y)


def save_predictions(filename, y):
    """
    Dumps y into .npy file
    """
    np.save(filename, y)# Load the data

