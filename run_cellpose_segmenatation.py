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


# load images from mix library

path_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix\selected_images'
os.chdir(path_norm)
images_name = glob.glob("*.tif")

AIPS_pose_object = AC.AIPS_cellpose(Image_name = images_name[0], path= path_norm, model_type="cyto", channels=[0,0])
img = AIPS_pose_object.cellpose_image_load()
# create mask for the entire image
mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0,:,:])

table["prediction"] = 'Na'
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
        table.loc[i,"prediction"] = pred








os.chdir(r'F:\HAB_2\PrinzScreen\training_classfication\models')
cnn_transfer_learning_Augmentation_drop_layer_4and5 = load_model('cnn_transfer_learning_Augmentation_drop_layer_4and5.h5')
predictions = cnn_transfer_learning_Augmentation_drop_layer_4and5.predict(test, verbose=0)










import numpy as np
from scipy.stats import skew
from skimage import data, util
from skimage import measure


def sd_intensity(regionmask, intensity_image):
    return np.std(intensity_image[regionmask])


def skew_intensity(regionmask, intensity_image):
    return skew(intensity_image[regionmask])

def pixelcount(regionmask):
    return np.sum(regionmask)

def mean_int(regionmask, intensity_image):
    return np.mean(intensity_image[regionmask])



prop_names = [
    "label",
    "area",
    "eccentricity",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "moments",
    "moments_central",
    "moments_hu",
    "moments_normalized",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    # "slice",
    "solidity"
]
table_prop = measure.regionprops_table(mask, img[0,:100,:100], properties=prop_names, extra_properties=(sd_intensity, skew_intensity,pixelcount , mean_int) )
tesdt = pd.DataFrame(table_prop)





# create mask for first label
stack,stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input = img[0,:,:], extract_pixel = 50, resize_pixel = 150, img_label = table.index.values[0])

#display compsite with outline
# comp_img = AFD.Compsite_display(input_image = img[0,:200,:200] , mask_roi = mask[0:200,0:200]).draw_ROI_contour()
# plt.imshow(comp_img)

i=23
print("stack {}  image image name {}".format(i, images_name[0]))

table.to_csv("test.csv")

