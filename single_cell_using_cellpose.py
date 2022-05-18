'''
Break multi cell image into single cell for generating data for machine learning
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
import glob
import tifffile as tfi
import random
import string
import re
import pandas as pd

from utils import AIPS_cellpose as AC
from utils import AIPS_file_display as AFD



def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    # print random string
    return result_str


path_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\norm'
path_mix = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix'
path_pheno =  r'F:\HAB_2\PrinzScreen\training_classfication\raw\pheno'

path_out_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\norm\SC'
path_out_mix = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix\SC'
path_out_pheno =  r'F:\HAB_2\PrinzScreen\training_classfication\raw\pheno\SC'



def image_to_cell(path,label,path_out):
    '''
    :param path: str
    :param label: str
    saves stack of singel cell image
    '''
    os.chdir(path)
    images_name = glob.glob("*.tif")
    for z in range(0,len(images_name)):
        print ("z: {}".format(z))
        os.chdir(path)
        AIPS_pose_object = AC.AIPS_cellpose(Image_name=images_name[z], path=path, model_type="cyto",channels=[0, 0])
        img = AIPS_pose_object.cellpose_image_load()
        # create mask for the entire image
        mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0, :, :])
        table_feature = AIPS_pose_object.measure_properties(input_image=img[0, :, :])
        if len(table) > 2:
            #save table
            table_feature = pd.DataFrame(table_feature)
            table_feature['class'] = label
            os.chdir(path)
            # table_name = re.sub(".tif", "", images_name[z])
            # filename1 = datetime.now().strftime("%Y%m%d_%H%M%S")
            # table_feature.to_csv(label + get_random_string(8) + filename1 + table_name + ".csv")
            # create mask for first label
            for i in range(len(table)):
                stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0, :, :], extract_pixel=50,resize_pixel=150, img_label=table.index.values[i])
                if stack is None:
                    continue
                else:
                    filename1 = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.chdir(path_out)
                    img_name = re.sub(".tif", "", images_name[z])
                    plt.imsave(label + get_random_string(8) + filename1 + img_name + ".png", stack)
                    #time.sleep(2)
                    print("stack {}  image image name {} -------- {}".format(i,images_name[z],z))
        else:
            continue
#
# z# # start with mix

image_to_cell(path = path_norm,label ='norm_',path_out = path_out_norm)
image_to_cell(path = path_pheno,label ='pheno_',path_out = path_out_pheno)

