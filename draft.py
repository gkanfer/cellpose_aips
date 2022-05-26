import pymc3 as pm
import theano.tensor as tt
from scipy import stats
from scipy.special import expit as logistic
from scipy.special import softmax
import arviz as az

import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib as mpl
from urllib.parse import urlparse
import tqdm
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

