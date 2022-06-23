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
from skimage.measure import label, regionprops
plt.imshow(label(img_gs))


if os.path.exists(os.path.join(path_out, 'binary.tif')):
    os.remove(os.path.join(path_out, 'binary.tif'))
tfi.imsave(os.path.join(path_out, 'binary.tif'), img_gs)