'''
AIPS program for calling pex phenotype
Segmentation: cell pose
calling: transfer learning VGG16 with augmentation


NIS-elementas outproc command:

@echo on
call D:\Gil\anaconda_gil\Scripts\activate.bat
call activate py37
call python D:\Gil\AIPS\Activate_nuclus\example_image\run_aips_batch.py
@pause

'''
import time, os, sys
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tfi
import random
from utils import AIPS_cellpose as AC

file = 'input.tif'
path_input = r'D:\Gil\AIPS\Activate_nuclus\example_image'
#path_out = r'D:\Gil\AIPS\Activate_nuclus\binary'
path_out = path_input

#upload model
AIPS_pose_object = AC.AIPS_cellpose(Image_name = file, path= path_input, model_type="cyto", channels=[0,0])
img = AIPS_pose_object.cellpose_image_load()

# create mask for the entire image
mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img)
tfi.imread(os.path.join(path_input,file))

table["predict"] = 0.7
table_na_rmv = table.sample(frac=1)[:5]

image_blank = np.zeros_like(img)
binary, table_sel = AIPS_pose_object.call_bin(table_sel_cor = table_na_rmv, threshold = 0.5 ,img_blank = image_blank)
from skimage import io, filters, measure, color, img_as_ubyte
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
import matplotlib.pyplot as plt
#plt.imshow(binary)
img_gs = img_as_ubyte(binary)
# from skimage.measure import label, regionprops
# img_gs = label(img_gs)
if os.path.exists(os.path.join(path_out, 'binary.tif')):
    os.remove(os.path.join(path_out, 'binary.tif'))
tfi.imsave(os.path.join(path_out, 'binary.tif'), img_gs)

with open(os.path.join(path_out, 'cell_count.txt'), 'r') as f:
    prev_number = f.readlines()
new_value = int(prev_number[0]) + len(table_na_rmv)
with open(os.path.join(path_out, 'cell_count.txt'), 'w') as f:
    f.write(str(new_value))


#
# if os.path.exists(os.path.join(path_out, 'binary.jpg')):
#     os.remove(os.path.join(path_out, 'binary.jpg'))
# PIL_image = Image.fromarray(img_gs)
# plt.imshow(PIL_image)
# plt.imsave(os.path.join(path_out, 'binary.jpg'),PIL_image)

#
# if os.path.exists(os.path.join(path_out, 'binary.tif')):
#     os.remove(os.path.join(path_out, 'binary.tif'))
# with open('binary.tif', 'w') as f:
#     tfi.imsave(os.path.join(path_out, 'binary.tif'), binary)
