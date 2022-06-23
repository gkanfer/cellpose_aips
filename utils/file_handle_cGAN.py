from skimage.color import rgba2rgb
import numpy as np
from random import randint
import os
import glob
import tensorflow as tf
import time
import glob
import tifffile as tfi

from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import pandas as pd
from skimage import io
from skimage.morphology import label

def i16_to_Gray_3ch(img_name=None,img=None):
    if img is None:
        img_temp = tfi.imread(img_name)
    else:
        img_temp = img
    input_gs_image = (img_temp / img_temp.max()) * 255
    ch2_u8 = np.uint8(input_gs_image)
    rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = ch2_u8
    rgb_input_img[:, :, 1] = ch2_u8
    rgb_input_img[:, :, 2] = ch2_u8
    return rgb_input_img

def mask_3ch_label(img_name=None,img=None):
    if img is None:
        img_temp = tfi.imread(img_name)
    else:
        img_temp = img
    #img_temp = img_temp[:, :, 0]
    rgb_input_img = np.zeros((np.shape(img_temp)[0], np.shape(img_temp)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = label(img_temp)
    rgb_input_img[:, :, 1] = label(img_temp)
    rgb_input_img[:, :, 2] = label(img_temp)
    return rgb_input_img


def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

def file_name_list_win(path, extanstion = '*\\*jpg'):
    '''
    list of  file names and the path
    for windows only
    '''
    file_names = []
    for files in glob.glob(path + extanstion):
        file_names.append(files)
    return file_names



