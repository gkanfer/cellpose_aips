'''
AIPS program for calling pex phenotype
Segmentation: cell pose
calling: transfer learning VGG16 with augmentation
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

if (__name__ == "__main__"):
    import argparse
    parser = argparse.ArgumentParser(description='AIPS activation')
    parser.add_argument('--file', dest='file', type=str, required=True,
                        help="The name of the image to analyze")
    parser.add_argument('--path', dest='path', type=str, required=True,
                        help="The path to the file")
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help="The name of the model")
    parser.add_argument('--path_model', dest='path_model', type=str, required=True,
                        help="The path to the model")
    args = parser.parse_args()
    #upload model
    model_cnn  = load_model(os.path.join(args.path_model,args.model))
    AIPS_pose_object = AC.AIPS_cellpose(Image_name = args.file, path= args.path, model_type="cyto", channels=[0,0])
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
            pred = model_cnn.predict(test_imgs_scaled.reshape(1, 150, 150, 3),verbose=0).tolist()[0][0]
            table.loc[i,"predict"] = pred
    # remove nas
    table_na_rmv = table.loc[table['predict']!='Na',:]
    #
    # predict = np.where(table_na_rmv.loc[:,'predict'] > 0.5, 1, 0)
    # table_na_rmv = table_na_rmv.assign(predict = predict)
    # remove small area
    table_na_rmv = table_na_rmv.loc[table['area'] > 1500,:]