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
import random


path = r'F:\HAB_2\PrinzScreen\training_classfication\traindata\test_data'
train_files = glob.glob(os.path.join(path, '*.' + 'png'))
train_files_new = []
total_cell_number = len(train_files)
norm = 3
phon = 500
random.shuffle(train_files)
count_pheno = 0
count_norm = 0
for i in range(len(train_files)):
    if train_files[i].split('\\')[6].split('_')[0].strip()=='norm':
            if count_norm < norm:
                train_files_new.append(train_files[i])
                count_norm += 1
    else:
        train_files_new.append(train_files[i])
train_files = train_files_new # update variable



def upload_image_list(path,itr_number):
    train_files = glob.glob(os.path.join(path, '*.' + 'png'))









path_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\mix\selected_images'
os.chdir(path_norm)
images_name = glob.glob("*.tif")

AIPS_pose_object = AC.AIPS_cellpose(Image_name = images_name[1], path= path_norm, model_type="cyto", channels=[0,0])
img = AIPS_pose_object.cellpose_image_load()
# create mask for the entire image
mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0,:,:])
plt.imshow(mask)



d = pm.Normal.dist(mu=0, sd=1)
d.dist()
d.random(size=10)
d.logp(0).eval()

RANDOM_SEED = 8927
np.random.seed(286)
path_origen = r'F:\HAB_2\PrinzScreen\training_classfication\raw'
path_norm = r'F:\HAB_2\PrinzScreen\training_classfication\raw\norm'
path_pheno = r'F:\HAB_2\PrinzScreen\training_classfication\raw\pheno'
train_dir = os.path.join(path_origen, 'training_data')
test_dir = os.path.join(path_origen, 'test_data')
# norm merage
def merge_cvs(path):
    file_names = [file for file in os.listdir(path) if file.endswith('csv')]
    df = pd.DataFrame()
    os.chdir(path)
    for i in tqdm(range(len(file_names))):
        df = pd.concat([df, pd.read_csv(file_names[i])], ignore_index=True)
    return df

df_norm = merge_cvs(path_norm)
df_pheno = merge_cvs(path_pheno)


def standardize(series):
    """Standardize a pandas series"""
    return (series - series.mean()) / series.std()

df = pd.concat([df_norm, df_pheno], ignore_index=True)

from sklearn.utils import shuffle
df = shuffle(df).reset_index()
df.head()

class_name =  df['class'].values
le = preprocessing.LabelEncoder()
class_name_encoded=le.fit_transform(class_name).tolist()
class_name_encoded[:5]

table_summary_id = df
# remove outliers



table_dict={'id':table_summary_id.index.to_list(),
            'Ecc_std':standardize(table_summary_id.eccentricity),
            'ex_std':standardize(table_summary_id.extent),
            'orien_std':standardize(table_summary_id.orientation),
            'solid_std':standardize(table_summary_id.solidity),
            'feret_diameter_max':standardize(np.log(table_summary_id.feret_diameter_max).values),
            'peri_log':standardize(np.log(table_summary_id.perimeter).values),
            'area_log':standardize(np.log(table_summary_id.area).values),
            'class_label':class_name_encoded}
table = pd.DataFrame(table_dict)
table = table.loc[table.area_log > -2.5,:]
table_complete = table.loc[:24999,:]


from numpy.ma.core import shape


print('The lenght of the input tasble is {}'.format(len(table)))

#chose the first 60
table = table_complete.loc[:500,:]
# table test 100 observation
table_test = table_complete.loc[500:599,:]

# we will try 3 models
id = np.array(table.id)
Ecc_std = np.array(table.Ecc_std)
ex_std = np.array(table.ex_std)
orien_std = np.array(table.orien_std)
solid_std = np.array(table.solid_std)
peri_log = np.array(table.peri_log)
area_log = np.array(table.area_log)

nid = len(np.unique(table.id))
if __name__ == '__main__':
    with pm.Model() as flat_proir:
        a = pm.Normal("a", 0, 1)
        b = pm.Normal("b", 0, 1)
        c = pm.Normal("c", 0, 1)
        d = pm.Normal("d",0,1)
        e = pm.Normal("e",0,1)
        f = pm.Normal("f",0,1)
        g = pm.Normal("g",0,1)
        Ecc = pm.Data("Ecc",Ecc_std)
        ex = pm.Data("ex",ex_std)
        orien = pm.Data("orien",orien_std)
        solid = pm.Data("solid",solid_std)
        peri = pm.Data("peri",peri_log)
        area = pm.Data("area",area_log)
        p = pm.Deterministic("p", pm.invlogit(a + b*Ecc + c*ex + d*orien + e*solid + f*peri + g*area))
        data = pm.Data("data", table)
        class_label = pm.Binomial("class_label", 1, p, observed=np.array(table.class_label))
        trace_flat = pm.sample(4000,tune=4000, random_seed=RANDOM_SEED,cores=1)
        idata_flat = az.from_pymc3(trace_flat)