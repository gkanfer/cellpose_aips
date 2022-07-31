'''
Load 80 kernel model
predict on 5 karnel
use the parameters from trace for creating a prediction model
'''

import pandas as pd
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.max_rows', 500)
import pickle

from jupyter_dash import JupyterDash
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
import seaborn as sns
sns.set()
import arviz as az
import pymc3 as pm
print(pm.__version__)
import theano.tensor as tt
import patsy

import os
import re
import glob
import random
import plotnine
from sklearn import preprocessing
from tqdm import tqdm

import plotly.express as px

from skimage import measure, restoration,morphology
from skimage import io, filters, measure, color, img_as_ubyte
from skimage.draw import disk
from skimage import measure, restoration,morphology

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)

os.chdir(r'F:\HAB_2\PrinzScreen\training_classfication')
from utils import AIPS_cellpose as AC
from utils import AIPS_file_display as AFD


# Load trace_80_Kernel
path_output_train_simple_model = r'F:\HAB_2\PrinzScreen\output_lab_meeting\Kerenl_80_image_0_train'
path_output_test_simple_model = r'F:\HAB_2\PrinzScreen\output_lab_meeting\Kerenl_5_image_1_test'
with open(os.path.join(path_output_train_simple_model,'simple_model_decay_Image_0_80kernel.pkl'), 'rb') as buff:
    data = pickle.load(buff)
trace_80_Kernel = data['trace']
# load model
def model_factory(x, y, z, sig):
    with pm.Model() as model:
        a = pm.Normal('a', 0.05, 0.1, shape=y)
        b = pm.Exponential('b', 0.1, shape=y)
        c = pm.Normal('c', 0.5, 0.1, shape=y)

        mu = a[z] + c[z] * tt.exp(-b[z] * x)  # linear model
        sigma_within = pm.Exponential("sigma_within", 1.0)  # prior stddev within image
        signal = pm.Normal("signal", mu=mu, sigma=sigma_within, observed=sig)  # likelihood
    return model

####### load test data ###########

table_img_test = pd.read_csv(os.path.join(path_output_test_simple_model,'table_img_test.csv'))
table_test = pd.read_csv(os.path.join(path_output_test_simple_model,'table_test.csv'))

###### predict with 5 kernel
aa = np.array(table_img_test['image_group'])
_, idx__ = np.unique(aa, return_index=True)
id_test = np.array(np.repeat(aa[np.sort(idx__)],5))
opening_opr__test = np.array(table_img_test.raius_list.values) # 55*20
opening_opr__test = opening_opr__test + 1
signal__test = np.array(table_img_test.image_signal.values) # 55*20
id_paramaters__test = len(idx__)

print('Number of observations:{}'.format(id_paramaters__test))
print('id:{}'.format(len(id_test)))
print('opening_opr:{}'.format(len(opening_opr__test)))
print('signal:{}'.format(len(signal__test)))

with model_factory(x=opening_opr__test,
                   y=id_paramaters__test,
                   z=id_test,
                   sig=signal__test) as test_model:
    # We first have to extract the learnt global effect from the train_trace
    df = pm.trace_to_dataframe(trace_80_Kernel,
                               varnames=["a","b","c",'sigma_within'],
                               include_transformed=True)
    # We have to supply the samples kwarg because it cannot be inferred if the
    # input trace is not a MultiTrace instance
    p_post = pm.sample_posterior_predictive(trace=df.to_dict('records'),var_names=["a","b","c",'sigma_within'],samples=4000)

################################################ prediction table
b = np.mean(p_post['b'],0)
b_sd = np.std(p_post['b'],0)
c = np.mean(p_post['c'],0)
c_sd = np.std(p_post['c'],0)
a = np.mean(p_post['a'],0)
a_sd = np.std(p_post['a'],0)
uniqe_list_image_name = np.unique(table_img_test['image_group'])

df_pred = pd.DataFrame({'a':a,'b':b,'c':c,'a_sd':a_sd,'b_sd':b_sd,"c_sd":c_sd,'Image':uniqe_list_image_name})
table_pred = pd.concat((table_test,df_pred),1)
table_pred

############################################## Peroxisome classification using Gaussian process follow prediction
# Distance matrix
table_micron = table_test
from scipy.spatial import distance_matrix
# table change nano distance to micron
table_micron.loc[:,['centroid-0','centroid-1']] = (table_micron.loc[:,['centroid-0','centroid-1']])/1000
#df = pd.DataFrame(table_micron, columns=['centroid-0', 'centroid-1'], index=table_micron.index.values.tolist())
df = pd.DataFrame(table_micron, columns=['centroid-0', 'centroid-1'])
dmat = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
#dmat = dmat.iloc[:66,:]
dmat = np.array(dmat)
dmat = dmat + 0.000000001
dmat
id_arr_uniqe = np.unique(table_img_test.image_group)
a_pred = table_pred.a.values
b_pred = table_pred.b.values
c_pred = table_pred.c.values

with pm.Model() as model_a:
    etasq = pm.Exponential("etasq", 0.5)
    ls_inv = pm.Exponential("ls_inv", 1)
    cov = etasq ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls_inv=ls_inv)
    gp = pm.gp.Latent(cov_func=cov)
    K = gp.prior("K", X=dmat)
    p = pm.Deterministic('p', tt.exp(a*K[id_arr_uniqe]) / (1. + tt.exp(a*K[id_arr_uniqe])))
    # likelihood
    trace_a = pm.sample(4000, tune=4000, cores=1, random_seed=RANDOM_SEED, target_accept=0.9, return_inferencedata=False)

# prediction table



def model_factory(para,ind_arr, mat):
    with pm.Model() as model:
        etasq = pm.Exponential("etasq", 0.5)
        ls_inv = pm.Exponential("ls_inv", 1)
        cov = etasq ** 2 * pm.gp.cov.ExpQuad(input_dim=1, ls_inv=ls_inv)
        gp = pm.gp.Latent(cov_func=cov)
        K = gp.prior("K", X=mat)
        p = pm.Deterministic('p', tt.exp(para * K[ind_arr]) / (1. + tt.exp(para * K[ind_arr])))
    return model



with model_factory(para = a_pred,
                   ind_arr = id_arr_uniqe,
                   mat = dmat) as train_model:
    train_trace_a = pm.sample(4000, tune=4000, target_accept=0.9,random_seed=RANDOM_SEED)


with model_factory(para = b_pred,
                   ind_arr = id_arr_uniqe,
                   mat = dmat) as train_model:
    train_trace_b = pm.sample(4000, tune=4000, target_accept=0.9,random_seed=RANDOM_SEED)


with model_factory(para = c_pred,
                   ind_arr = id_arr_uniqe,
                   mat = dmat) as train_model:
    train_trace_c = pm.sample(4000, tune=4000, target_accept=0.9,random_seed=RANDOM_SEED)


p_a = np.mean(train_trace_a['p'],0)
p_b = np.mean(train_trace_b['p'],0)
p_c = np.mean(train_trace_c['p'],0)
