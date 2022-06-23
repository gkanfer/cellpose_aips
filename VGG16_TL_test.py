'''
050222

3 model were generated from Heather data
1) cellpose for segmentation
2) model for prediction
3) retain the phenotipic cells
5) show prediction on images
4) Make a roc curve
'''

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
from datetime import datetime
from cellpose.io import logger_setup
from cellpose import utils
import tensorflow as tf
from tensorflow import keras
from skimage import io, filters, measure, color, img_as_ubyte
import glob
import tifffile as tfi
import random
import string
import re

from utils import AIPS_cellpose as AC
from utils import AIPS_file_display as AFD
from utils import display_and_xml as dix

print(keras.__version__)
print(tf.__version__)

# path to mixed images
path_mix = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix'
# load TF models

basic =  tf.keras.models.load_model(os.path.join(r'F:\HAB_2\PrinzScreen\training_classfication\models','pex_basic.h5'))
no_aug_tfl =  tf.keras.models.load_model(os.path.join(r'F:\HAB_2\PrinzScreen\training_classfication\models','pex_aug_dropout.h5'))
aug_tfl = tf.keras.models.load_model(os.path.join(r'F:\HAB_2\PrinzScreen\training_classfication\models','pex_aug_dropout_vgg16_unfreezblock_4and5.h5'))
hether_model =  tf.keras.models.load_model(os.path.join(r'F:\HAB_2\PrinzScreen\training_classfication\models','prinzCAT.h5'))

AIPS_pose_object = AC.AIPS_cellpose(Image_name='exp001_13DKO_1-3.tif', path=path_mix, model_type="cyto", channels=[0, 0])
img = AIPS_pose_object.cellpose_image_load()
mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0,:,:])
table['predict'] = -1

for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0, :, :],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        plt.imsave("temp.png",stack)
        img_inp = skimage.io.imread("test.png")
        x = tf.expand_dims(img_inp[:, :, :3], 0)
        predictions = basic.predict(x, verbose=0)
        table.loc[i,'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i,'predict']))



info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,:,:],prediction_table=table,font_select="arial.ttf",font_size=20,windows=True)
plt.imsave('base_img.png',PIL_image)
# plt.imshow(PIL_image,cmap='gray')
#
compsite_object = AFD.Compsite_display(input_image=img[0,:,:],mask_roi=mask)
img_comp = compsite_object.draw_ROI_contour()
plt.imsave('base_img_comp.png',img_comp)
# plt.im(img_comp)

table['predict'] = -1

for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0, :, :],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        rgb_input_img = np.zeros((np.shape(stack)[0], np.shape(stack)[1], 3), dtype=np.float64)
        rgb_input_img[:, :, 0] = stack
        rgb_input_img[:, :, 1] = stack
        rgb_input_img[:, :, 2] = stack
        rgb_input_img = rgb_input_img / rgb_input_img.max()
        rgb_input_img = rgb_input_img * 2 ** 8
        rgb_input_img = rgb_input_img.astype(np.uint8)
        x = tf.expand_dims(rgb_input_img, 0)
        predictions = no_aug_tfl.predict(x, verbose=0)
        table.loc[i,'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i,'predict']))

info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,:,:],prediction_table=table,font_select="arial.ttf",font_size=24,windows=True)
plt.imsave('no_aug_tfl_img.png',PIL_image)

compsite_object = AFD.Compsite_display(input_image=img[0,:,:],mask_roi=mask)
img_comp = compsite_object.draw_ROI_contour()
plt.imsave('no_aug_tfl_img_comp.png',img_comp)

###############################################################


table['predict'] = -1

for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0, :, :],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        rgb_input_img = np.zeros((np.shape(stack)[0], np.shape(stack)[1], 3), dtype=np.float64)
        rgb_input_img[:, :, 0] = stack
        rgb_input_img[:, :, 1] = stack
        rgb_input_img[:, :, 2] = stack
        rgb_input_img = rgb_input_img / rgb_input_img.max()
        rgb_input_img = rgb_input_img * 2 ** 8
        rgb_input_img = rgb_input_img.astype(np.uint8)
        x = tf.expand_dims(rgb_input_img, 0)
        predictions = aug_tfl.predict(x, verbose=0)
        table.loc[i,'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i,'predict']))

info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,:,:],prediction_table=table,font_select="arial.ttf",font_size=24,windows=True)
plt.imsave('aug_tfl_img.png',PIL_image)

compsite_object = AFD.Compsite_display(input_image=img[0,:,:],mask_roi=mask)
img_comp = compsite_object.draw_ROI_contour()
plt.imsave('no_aug_tfl_img_comp.png',img_comp)




'''

test heather model on subset of images

'''


################################# testing hether model


# try to focus on phnotype

AIPS_pose_object = AC.AIPS_cellpose(Image_name='exp001_13DKO_1-3.tif', path=path_mix, model_type="cyto", channels=[0, 0])
img = AIPS_pose_object.cellpose_image_load()

plt.imshow(img[0,1700:2100,1100:1500])

mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0,1700:2100,1100:1500])
table['predict'] = -1

for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,1700:2100,1100:1500],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        plt.imsave("temp.png",stack)
        img_inp = skimage.io.imread("test.png")
        x = tf.expand_dims(img_inp[:, :, :1], 0)
        predictions = hether_model.predict(x, verbose=0)
        table.loc[i,'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i,'predict']))



info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,1700:2100,1100:1500],prediction_table=table,font_select="arial.ttf",font_size=20,windows=True)
plt.imsave('heather_img.png',PIL_image)

img_inp = skimage.io.imread("test.png")
x = tf.expand_dims(img_inp[:, :, :1], 0)
predictions = hether_model.predict(x, verbose=0)


# does not work


AIPS_pose_object = AC.AIPS_cellpose(Image_name='exp001_13DKO_1-3.tif', path=path_mix, model_type="cyto", channels=[0, 0])
img = AIPS_pose_object.cellpose_image_load()

plt.imshow(img[0,1700:2100,1100:1500])

mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0,1700:2100,1100:1500])

# base
table['predict'] = -1
for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,1700:2100,1100:1500],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        plt.imsave("temp.png",stack)
        img_inp = skimage.io.imread("test.png")
        x = tf.expand_dims(img_inp[:, :, :3], 0)
        predictions =basic.predict(x, verbose=0)
        table.loc[i,'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i,'predict']))



info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,1700:2100,1100:1500],prediction_table=table,font_select="arial.ttf",font_size=20,windows=True)
plt.imsave('basic_img.png',PIL_image)

#no aU

table['predict'] = -1
for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,1700:2100,1100:1500],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        plt.imsave("temp.png",stack)
        img_inp = skimage.io.imread("test.png")
        x = tf.expand_dims(img_inp[:, :, :3], 0)
        predictions = no_aug_tfl.predict(x, verbose=0)
        table.loc[i,'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i,'predict']))



info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,1700:2100,1100:1500],prediction_table=table,font_select="arial.ttf",font_size=20,windows=True)
plt.imsave('no_aug_tfl_img.png',PIL_image)

# ou

table['predict'] = -1

for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,1700:2100,1100:1500],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        plt.imsave("temp.png",stack)
        img_inp = skimage.io.imread("test.png")
        x = tf.expand_dims(img_inp[:, :, :3], 0)
        predictions = aug_tfl.predict(x, verbose=0)
        table.loc[i,'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i,'predict']))


info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,1700:2100,1100:1500],prediction_table=table,font_select="arial.ttf",font_size=20,windows=True)
plt.imsave('au_tl_img.png',PIL_image)

compsite_object = AFD.Compsite_display(input_image=img[0,1700:2100,1100:1500],mask_roi=mask)
img_comp = compsite_object.draw_ROI_contour()
plt.imshow(img_comp)
info_table, PIL_image = AIPS_pose_object.display_image_prediction(img =img_comp,prediction_table=table,font_select="arial.ttf",font_size=20,windows=True)



'''

test models without saving png 

'''

AIPS_pose_object = AC.AIPS_cellpose(Image_name='exp001_13DKO_1-3.tif', path=path_mix, model_type="cyto", channels=[0, 0])
img = AIPS_pose_object.cellpose_image_load()

#plt.imshow(img[0,1700:2100,1100:1500])

mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0,1700:2100,1100:1500])

# base
table['predict'] = -1
for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,1700:2100,1100:1500],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        rgb_input_img = np.zeros((np.shape(stack)[0], np.shape(stack)[1], 3), dtype=np.float64)
        rgb_input_img[:, :, 0] = stack
        rgb_input_img[:, :, 1] = stack
        rgb_input_img[:, :, 2] = stack
        rgb_input_img = rgb_input_img / rgb_input_img.max()
        rgb_input_img = rgb_input_img * 2 ** 8
        rgb_input_img = rgb_input_img.astype(np.uint8)
        x = tf.expand_dims(rgb_input_img, 0)
        predictions = basic.predict(x, verbose=0)
        table.loc[i, 'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i, 'predict']))



info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,1700:2100,1100:1500],prediction_table=table,font_select="arial.ttf",font_size=20,windows=True)
plt.imsave('basic_img.png',PIL_image)

#no aU

table['predict'] = -1
for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,1700:2100,1100:1500],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        rgb_input_img = np.zeros((np.shape(stack)[0], np.shape(stack)[1], 3), dtype=np.float64)
        rgb_input_img[:, :, 0] = stack
        rgb_input_img[:, :, 1] = stack
        rgb_input_img[:, :, 2] = stack
        rgb_input_img = rgb_input_img / rgb_input_img.max()
        rgb_input_img = rgb_input_img * 2 ** 8
        rgb_input_img = rgb_input_img.astype(np.uint8)
        x = tf.expand_dims(rgb_input_img, 0)
        predictions = no_aug_tfl.predict(x, verbose=0)
        table.loc[i, 'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i, 'predict']))



info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,1700:2100,1100:1500],prediction_table=table,font_select="arial.ttf",font_size=20,windows=True)
plt.imsave('no_au_tl_img.png',PIL_image)

# ou

table['predict'] = -1

for i in range(0,len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,1700:2100,1100:1500],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        print('stack is none')
        continue
    else:
        print('run number{}'.format(str(table.index.values[i])))
        rgb_input_img = np.zeros((np.shape(stack)[0], np.shape(stack)[1], 3), dtype=np.float64)
        rgb_input_img[:, :, 0] = stack
        rgb_input_img[:, :, 1] = stack
        rgb_input_img[:, :, 2] = stack
        rgb_input_img = rgb_input_img / rgb_input_img.max()
        rgb_input_img = rgb_input_img * 2 ** 8
        rgb_input_img = rgb_input_img.astype(np.uint8)
        x = tf.expand_dims(rgb_input_img, 0)
        predictions = aug_tfl.predict(x, verbose=0)
        table.loc[i, 'predict'] = predictions.tolist()[0][0]
        print('{}'.format(table.loc[i, 'predict']))

info_table, PIL_image = AIPS_pose_object.display_image_prediction(img = img[0,1700:2100,1100:1500],prediction_table=table,font_select="arial.ttf",font_size=20,windows=True)
plt.imsave('au_tl_img.png',PIL_image)


stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,1700:2100,1100:1500],
                                                                                     extract_pixel=50,
                                                                                     resize_pixel=150,
                                                                                     img_label=table.index.values[32])
rgb_input_img = np.zeros((np.shape(stack)[0], np.shape(stack)[1], 3), dtype=np.float64)
rgb_input_img[:, :, 0] = stack
rgb_input_img[:, :, 1] = stack
rgb_input_img[:, :, 2] = stack
rgb_input_img = rgb_input_img / rgb_input_img.max()
rgb_input_img = rgb_input_img * 2 ** 8
rgb_input_img = rgb_input_img.astype(np.uint8)
x = tf.expand_dims(rgb_input_img, 0)
predictions = aug_tfl.predict(x, verbose=0)
table.loc[i, 'predict'] = predictions.tolist()[0][0]

