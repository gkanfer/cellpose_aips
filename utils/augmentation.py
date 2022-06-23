'''
try to use global paramters such as 256 pixel

'''

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
from skimage.color import gray2rgb
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import gray2rgb
from skimage.morphology import label
import utils.file_handle_cGAN as fh

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

def random_crop(input_image, real_image,IMG_HEIGHT,IMG_WIDTH):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    normalized_input = 2 * input_image - 1
    normalized_real_image = 2 * real_image - 1
    return normalized_input, normalized_real_image

@tf.function()
def random_jitter(input_image, real_image):
    '''
        float input image returm jitter
    '''
    # Resizing up in 1.2 factor 286x286
    input_image, real_image = resize(input_image, real_image, 288,288)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image,256,256)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file):
    '''
        apply augmentation
    '''
    input_image, real_image = fh.load(image_file)
    input_image, real_image = random_jitter(input_image/255, real_image/255)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

def load_image_test(image_file):
    '''
            no augmentation
    '''
    input_image, real_image = fh.load(image_file)
    input_image, real_image = resize(input_image/255, real_image/255,256,256)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image