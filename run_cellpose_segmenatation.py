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

AIPS_pose_object = AC.AIPS_cellpose(Image_name = images_name[1], path= path_norm, model_type="cyto", channels=[0,0])
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
        plt.imsave("temp.png", stack)
        test_imgs = img_to_array(load_img("temp.png", target_size=(150, 150)))
        test_imgs = np.array(test_imgs)
        # test_imgs_scaled = test_imgs.astype('float32')
        test_imgs_scaled = test_imgs
        test_imgs_scaled /= 255
        pred = cnn_transfer_learning_Augmentation_drop_layer_4and5.predict(test_imgs_scaled.reshape(1, 150, 150, 3),
                                                                    verbose=0).tolist()[0][0]
        table.loc[i,"predict"] = pred

# remove nas
table_na_rmv = table.loc[table['predict']!='Na',:]
#
predict = np.where(table_na_rmv.loc[:,'predict'] > 0.5, 1, 0)
table_na_rmv = table_na_rmv.assign(predict = predict)
#remove small area
table_na_rmv = table_na_rmv.loc[table['area'] > 1500,:]



# get mask
compsite_object = AFD.Compsite_display(input_image=img[0,:,:],mask_roi=mask)
comp_img = compsite_object.draw_ROI_contour()
plt.imshow(comp_img)



table_pred, impil = AIPS_pose_object.display_image_prediction(img = comp_img ,prediction_table = table_na_rmv,
                                                              font_select = None, font_size = 14, windows=True,  lable_draw = 'predict',round_n = 3)
plt.imshow(impil)

## get only the mask of the postive structure
fig, ax = plt.subplots(1, 2, figsize=(26, 26))
ax[0].imshow(impil)
ax[1].imshow(mask)

# keep only target
table_na_rmv_trgt =table_na_rmv.loc[table_na_rmv['predict'] > 0.5,:]
x, y = table_na_rmv_trgt.loc[table_na_rmv_trgt.index[0], ["centroid-0", "centroid-1"]]
from skimage.draw import disk
img_mask = np.zeros((np.shape(img[0,:,:])[0],np.shape(img[0,:,:])[1]), dtype=np.uint8)
row, col = disk((int(y),int(x)), 20)
img_mask[row, col] = 1
fig, ax = plt.subplots(1, 3, figsize=(26, 26))
ax[0].imshow(impil)
ax[1].imshow(mask)
ax[2].imshow(img_mask)