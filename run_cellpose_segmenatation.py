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
from PIL import Image
from skimage import io, filters, measure, color, img_as_ubyte

from utils import AIPS_cellpose as AC
from utils import AIPS_file_display as AFD


'''
test VGG16 augmentation and including argumentation and transfer learning classification from 051722

############ cnn_transfer_learning_Augmentation_drop_layer_4and5 ##########################

'''
# laod model
os.chdir(r'F:\HAB_2\PrinzScreen\training_classfication\models')
cnn_transfer_learning_Augmentation_drop_layer_4and5 = load_model('cnn_transfer_learning_Augmentation_drop_layer_4and5.h5')

# load images from mix library

path_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix\selected_images'
os.chdir(path_norm)
images_name = glob.glob("*.tif")

AIPS_pose_object = AC.AIPS_cellpose(Image_name = images_name[0], path= path_norm, model_type="cyto", channels=[0,0])
img = AIPS_pose_object.cellpose_image_load()
# create mask for the entire image
mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0,:,:])

table["predict"] = 'Na'
test = []
for i in range(len(table)):
    stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,:,:],
                                                                                     extract_pixel=50, resize_pixel=150,
                                                                                     img_label=table.index.values[i])
    if stack is None:
        continue
    else:
        stack_8 = (stack / np.max(stack)) * 255
        stack_ub = stack_8.astype(np.uint8)
        im_pil = Image.fromarray(stack_ub).convert('RGB')
        test = []
        test.append(tf.convert_to_tensor(
            im_pil, dtype=np.float32, dtype_hint=None, name=None
        ))
        test = np.array(test)
        pred = cnn_transfer_learning_Augmentation_drop_layer_4and5.predict(test, verbose=0).tolist()[0][0]
        table.loc[i,"predict"] = pred

plt.imshow(im_pil)


# remove nas
table_na_rmv = table.loc[table['predict']!='Na',:]
#
predict = np.where(table_na_rmv.loc[:,'predict'] > 0, 1, 0)
table_na_rmv = table_na_rmv.assign(predict = predict)
# remove small area
table_na_rmv = table_na_rmv.loc[table['area'] > 3000,:]



# get mask
compsite_object = AFD.Compsite_display(input_image=img[0,:,:],mask_roi=mask)
comp_img = compsite_object.draw_ROI_contour()
plt.imshow(comp_img)



table_pred, impil = AIPS_pose_object.display_image_prediction(img = comp_img ,prediction_table = table_na_rmv,
                                                              font_select = None, font_size = 14, windows=True,  lable_draw = 'area',round_n = 30)
plt.imshow(impil)

stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0, :, :],
                                                                                 extract_pixel=50, resize_pixel=150,
                                                                                 img_label=table.index.values[42])
plt.imsave("test_ind42.png", stack)
test_imgs  = img_to_array(load_img("test_ind42.png", target_size=(150,150)))
test_imgs = np.array(test_imgs)
test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255
plt.imshow(test_imgs_scaled)

cnn_transfer_learning_Augmentation_drop_layer_4and5.predict(test_imgs_scaled.reshape(1,150,150,3), verbose=0).tolist()[0][0]



stack_8 = (stack / np.max(stack)) * 255
stack_ub = stack.astype(np.uint8)
im_pil =  np.array(Image.fromarray(stack_ub).convert('RGB'))
plt.imshow(im_pil)
test_imgs = np.array(im_pil)
test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255
plt.imshow(test_imgs_scaled)
cnn_transfer_learning_Augmentation_drop_layer_4and5.predict(test_imgs_scaled.reshape(1,150,150,3), verbose=0).tolist()[0][0]


test = []
test.append(tf.convert_to_tensor(
    im_pil, dtype=np.float32, dtype_hint=None, name=None
))
test = np.array(test)
pred = cnn_transfer_learning_Augmentation_drop_layer_4and5.predict(test, verbose=0).tolist()[0][0]