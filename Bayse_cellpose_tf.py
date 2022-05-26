import pymc3 as pm
import theano.tensor as tt
from scipy import stats
from scipy.special import expit as logistic
from scipy.special import softmax
from sklearn import preprocessing
import arviz as az

import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib as mpl
from urllib.parse import urlparse
import tqdm
from tqdm import tqdm
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
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
from skimage import io, filters, measure, color, img_as_ubyte

from utils import AIPS_cellpose as AC
from utils import AIPS_file_display as AFD


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
    #az.summary(idata_flat, round_to=2)

    with pm.Model() as flat_proir_condansed:
          a = pm.Normal("a", 0, 1)
          b = pm.Normal("b", 0, 0.38)
          c = pm.Normal("c", 0, 0.38)
          d = pm.Normal("d",0,0.38)
          e = pm.Normal("e",0,0.38)
          f = pm.Normal("f",0,0.2)
          g = pm.Normal("g",0,0.2)
          Ecc = pm.Data("Ecc",Ecc_std)
          ex = pm.Data("ex",ex_std)
          orien = pm.Data("orien",orien_std)
          solid = pm.Data("solid",solid_std)
          peri = pm.Data("peri",peri_log)
          area = pm.Data("area",area_log)
          p = pm.Deterministic("p", pm.invlogit(a + b*Ecc + c*ex + d*orien + e*solid + f*peri + g*area))
          data = pm.Data("data", table)
          class_label = pm.Binomial("class_label", 1, p, observed=np.array(table.class_label))
          trace_flat_condansed = pm.sample(4000,tune=4000, random_seed=RANDOM_SEED,cores=1)
          idata_flat_condansed = az.from_pymc3(trace_flat_condansed)

    # shrinking prioirs
    with pm.Model() as shrinking_priors:
        # hyper-priors
        a_bar = pm.Normal("a_bar", 0.0, 1)
        sigma_a = pm.Exponential("sigma_a", 0.1)
        sigma_b = pm.Exponential("sigma_b", 0.1)
        sigma_c = pm.Exponential("sigma_c", 0.1)
        sigma_d = pm.Exponential("sigma_d", 0.1)
        sigma_e = pm.Exponential("sigma_e", 0.1)
        sigma_f = pm.Exponential("sigma_f", 0.1)
        sigma_g = pm.Exponential("sigma_g", 0.1)
        # adaptive priors
        a = pm.Normal("a", a_bar, sigma_a)
        b = pm.Normal("b", 0, sigma_b)
        c = pm.Normal("c", 0, sigma_c)
        d = pm.Normal("d",0,sigma_d)
        e = pm.Normal("e",0,sigma_e)
        f = pm.Normal("f",0,sigma_f)
        g = pm.Normal("g",0,sigma_g)
        Ecc = pm.Data("Ecc",Ecc_std)
        ex = pm.Data("ex",ex_std)
        orien = pm.Data("orien",orien_std)
        solid = pm.Data("solid",solid_std)
        peri = pm.Data("peri",peri_log)
        area = pm.Data("area",area_log)
        p = pm.Deterministic("p", pm.invlogit(a + b*Ecc + c*ex + d*orien + e*solid + f*peri + g*area))
        data = pm.Data("data", table)
        class_label = pm.Binomial("class_label", 1, p, observed=np.array(table.class_label))
        trace_shrinking_priors = pm.sample(4000,tune=4000, random_seed=RANDOM_SEED,cores=1)
        idata_shrinking_priors = az.from_pymc3(trace_shrinking_priors)

    print('flat_proir')
    with flat_proir:
          pm.set_data({"Ecc":np.array(table_test.Ecc_std),
                        "ex":np.array(table_test.ex_std),
                        "orien":np.array(table_test.orien_std),
                        "solid":np.array(table_test.solid_std),
                        "peri":np.array(table_test.peri_log),
                        "area":np.array(table_test.area_log)})
          p_post = pm.sample_posterior_predictive(trace_flat, random_seed=RANDOM_SEED)

    p_test_pred = p_post['class_label'].mean(axis=0)
    table_test['predicted_trace_flat'] =  p_test_pred

    print('flat_proir_condansed')
    with flat_proir_condansed:
        pm.set_data({"Ecc":np.array(table_test.Ecc_std),
                    "ex":np.array(table_test.ex_std),
                    "orien":np.array(table_test.orien_std),
                    "solid":np.array(table_test.solid_std),
                    "peri":np.array(table_test.peri_log),
                    "area":np.array(table_test.area_log)})
        p_post = pm.sample_posterior_predictive(trace_flat_condansed, random_seed=RANDOM_SEED)

    p_test_pred = p_post['class_label'].mean(axis=0)
    table_test['predicted_trace_flat_condansed'] =  p_test_pred

    print('shrinking_priors')
    with shrinking_priors:
        pm.set_data({"Ecc":np.array(table_test.Ecc_std),
                    "ex":np.array(table_test.ex_std),
                    "orien":np.array(table_test.orien_std),
                    "solid":np.array(table_test.solid_std),
                    "peri":np.array(table_test.peri_log),
                    "area":np.array(table_test.area_log)})
        p_post = pm.sample_posterior_predictive(trace_shrinking_priors, random_seed=RANDOM_SEED)

    p_test_pred = p_post['class_label'].mean(axis=0)
    table_test['predicted_trace_shrinking_priors'] =  p_test_pred

# os.chdir(r'F:\HAB_2\PrinzScreen\data_bayes')
# table_test.to_csv('table_test_052622.csv',encoding='utf-8')

#### save model and trace

path_bayes_model = r'F:\HAB_2\PrinzScreen\model\bays'
import pickle

with open(os.path.join(path_bayes_model,'flat_proir.pkl'), 'wb') as buff:
    pickle.dump({'model': flat_proir, 'trace': trace_flat}, buff)


with open(os.path.join(path_bayes_model,'flat_proir_condansed.pkl'), 'wb') as buff:
    pickle.dump({'model': flat_proir_condansed, 'trace': trace_flat_condansed}, buff)


with open(os.path.join(path_bayes_model,'shrinking_priors.pkl'), 'wb') as buff:
    pickle.dump({'model': shrinking_priors, 'trace': trace_shrinking_priors}, buff)


# load and test pickle
path_bayes_model = r'F:\HAB_2\PrinzScreen\model\bays'
table_test = table_complete.loc[201:250,:]
with open(os.path.join(path_bayes_model,'shrinking_priors.pkl'), 'rb') as buff:
    data = pickle.load(buff)
model = data['model']
trace = data['trace']

with model:
    pm.set_data({"Ecc": np.array(table_test.Ecc_std),
                 "ex": np.array(table_test.ex_std),
                 "orien": np.array(table_test.orien_std),
                 "solid": np.array(table_test.solid_std),
                 "peri": np.array(table_test.peri_log),
                 "area": np.array(table_test.area_log)})
    p_post = pm.sample_posterior_predictive(trace, random_seed=RANDOM_SEED)

p_test_pred = p_post['class_label'].mean(axis=0)
table_test['predicted_trace_flat'] = p_test_pred
