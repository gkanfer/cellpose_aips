import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from urllib.parse import urlparse
from cellpose import models, core
use_GPU = core.use_gpu()
print('>>> GPU activated? %d'%use_GPU)
# call logger_setup to have output of cellpose written
from cellpose.io import logger_setup
from cellpose import utils
import tensorflow as tf
from tensorflow import keras
import glob
import pandas as pd
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
from skimage import io, filters, measure, color, img_as_ubyte


from utils import AIPS_cellpose as AC
from utils import AIPS_file_display as AFD

'''
Test two diffrent why of load data:
1) save png
2) convert to RGB
'''
path_input = r"F:\HAB_2\PrinzScreen\training_classfication\raw\mix\test_data"
os.chdir(r'F:\HAB_2\PrinzScreen\training_classfication\models')
Gil = load_model('cnn_transfer_learning_Augmentation_drop_layer_4and5.h5')


os.chdir(path_input)
images_name = glob.glob("*.png")

# gil prediction
for i in range(len(images_name)):
    os.chdir(path_input)
    test_imgs = img_to_array(load_img(images_name[i], target_size=(150, 150)))
    test_imgs = np.array(test_imgs)
    # test_imgs_scaled = test_imgs.astype('float32')
    test_imgs_scaled = test_imgs
    test_imgs_scaled /= 255
    pred = np.round(Gil.predict(test_imgs_scaled.reshape(1, 150, 150, 3),verbose=0).tolist()[0][0],2)
    img_gs = img_as_ubyte(test_imgs)
    PIL_image = Image.fromarray(img_gs)
    draw = ImageDraw.Draw(PIL_image)
    font = ImageFont.truetype("arial.ttf", 12, encoding="unic")
    draw.text((5, 5),str(pred), 'red', font=font)
    plt.subplot(4, 6, i + 1)
    plt.imshow(PIL_image)

# convert RGB
# for i in range(len(images_name)):
#     os.chdir(path_input)
#     test_imgs = img_to_array(load_img(images_name[i], target_size=(150, 150)))
#     test_imgs = np.array(test_imgs)
#     # test_imgs_scaled = test_imgs.astype('float32')
#     test_imgs_scaled = test_imgs
#     test_imgs_scaled /= 255
#     pred = np.round(Gil.predict(test_imgs_scaled.reshape(1, 150, 150, 3),verbose=0).tolist()[0][0],2)
#     img_gs = img_as_ubyte(test_imgs)
#     PIL_image = Image.fromarray(img_gs)
#     draw = ImageDraw.Draw(PIL_image)
#     font = ImageFont.truetype("arial.ttf", 12, encoding="unic")
#     draw.text((5, 5),str(pred), 'red', font=font)
#     plt.subplot(4, 6, i + 1)
#     plt.imshow(PIL_image)

