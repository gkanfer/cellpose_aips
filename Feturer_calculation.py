'''
Break multi cell image into single cell for generating data for machine learning
'''

import numpy as np
import time, os, sys
from urllib.parse import urlparse

import pandas as pd
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

from utils import AIPS_cellpose as AC
from utils import AIPS_file_display as AFD



def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    # print random string
    print(result_str)


path_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\norm'
path_mix = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix'
path_pheno =  r'F:\HAB_2\PrinzScreen\training_classfication\raw\pheno'

path_out_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\norm\SC'
path_out_mix = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix\SC'
path_out_pheno =  r'F:\HAB_2\PrinzScreen\training_classfication\raw\pheno\SC'



def image_to_table_properties(path,label,path_out):
    '''
    :param path: str
    :param label: str
    saves stack of singel cell image
    '''
    os.chdir(path)
    images_name = glob.glob("*.tif")
    for z in range(0,len(images_name)):
        print ("z: {}".format(z) )
        os.chdir(path)
        try:
            AIPS_pose_object = AC.AIPS_cellpose(Image_name=images_name[z], path=path, model_type="cyto",channels=[0, 0])
        except:
            continue
        try:
            img = AIPS_pose_object.cellpose_image_load()
        except:
            continue
        # create mask for the entire image
        try:
            mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0, :, :])
        except:
            continue
        try:
            table_feature = AIPS_pose_object.measure_properties(input_image=img[0, :, :])
        except:
            continue
        table_feature = pd.DataFrame(table_feature)
        table_feature['class'] = label
        os.chdir(path_out)
        table_name = re.sub(".tif", "", 'exp001_13DKO_1-4.tif')
        filename1 = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_feature.to_csv(label + str(get_random_string(8)) + filename1 + table_name + ".csv")
        print("stack {}  image image name {}".format(images_name[z],z))
#
# z# # start with mix
image_to_table_properties(path = path_mix,label ='mix_',path_out = path_mix)
image_to_table_properties(path = path_norm,label ='norm_',path_out = path_norm)
image_to_table_properties(path = path_pheno,label ='pheno_',path_out = path_pheno)
