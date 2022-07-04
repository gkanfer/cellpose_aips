import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
import skimage.io
from cellpose import models, core
use_GPU = core.use_gpu()
print('>>> GPU activated? %d'%use_GPU)


# from PIL import fromarray
from skimage.filters import threshold_local
from scipy.ndimage.morphology import binary_opening
from skimage import io, filters, measure, color, img_as_ubyte
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure, restoration,morphology
from skimage.exposure import rescale_intensity
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import re
import glob
import random
import plotnine
from sklearn import preprocessing
from tqdm import tqdm


from utils import AIPS_cellpose as AC
from utils import AIPS_file_display as AFD


path_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix\selected_images'
os.chdir(path_norm)
images_name = glob.glob("*.tif")

AIPS_pose_object = AC.AIPS_cellpose(Image_name = images_name[1], path= path_norm, model_type="cyto", channels=[0,0])
img = AIPS_pose_object.cellpose_image_load()
# create mask for the entire image
mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0,:,:])

table_gran_temp = AIPS_pose_object.measure_properties(input_image=img[0,:,:])
table_gran = pd.DataFrame(table_gran_temp['mean_int'].tolist())
open_vec = np.linspace(1, 80, 10, endpoint=False, dtype=int)
for i in range(1, len(open_vec)):
    selem = morphology.disk(open_vec[i], dtype=bool)
    eros_pix = morphology.erosion(img[0,:,:], selem=selem)
    rec = morphology.dilation(eros_pix, selem=selem)
    table_gran_temp = AIPS_pose_object.measure_properties(input_image=rec)
    table_gran[str(open_vec[i]) + '_mean_int'] = table_gran_temp['mean_int'].tolist()


# the differance between pex and SG

path_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix\selected_images'
os.chdir(path_norm)
images_name = glob.glob("*.tif")
image = skimage.io.imread(os.path.join(path_norm,images_name[1]))

path_tif = r'E:\Elliot\classification\data\tif\mix_ask_elliot'
os.chdir(path_tif)
images_name = glob.glob("*.tif")
image_sg = skimage.io.imread(os.path.join(path_tif,images_name[1]))


