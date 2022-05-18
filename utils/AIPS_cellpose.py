import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib as mpl
from urllib.parse import urlparse
from cellpose import models, core
from cellpose.io import logger_setup
from cellpose import utils
import glob
import pandas as pd
from scipy.stats import skew

from PIL import Image, ImageEnhance, ImageDraw,ImageFont
from skimage import io, filters, measure, color, img_as_ubyte

class AIPS_cellpose:
    def __init__(self, Image_name=None, path=None, image = None, mask = None, table = None, model_type = None, channels = None ):
        '''
        :param Image_name: str
        :param path: str
        :param image: inputimage for segmantion
        :param model_type: 'cyto' or model_type='nuclei'
        :param channels: # channels = [0,0] # IF YOU HAVE GRAYSCALE
                    channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
                    channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

                    or if you have different types of channels in each image
                    channels = [[2,3], [0,0], [0,0]]
                    channels = [1,1]
        '''
        self.Image_name = Image_name
        self.path = path
        self.image = image
        self.mask =  mask
        self.table = table
        self.model_type = model_type
        self.channels = channels

    def cellpose_image_load(self):
        ''':parameter
        Image: File name (tif format) - should be greyscale
        path: path to the file
        :return
        grayscale_image_container: dictionary of np array
        '''
        self.image = skimage.io.imread(os.path.join(self.path,self.Image_name))
        return self.image

    def cellpose_segmantation(self, image_input):
        use_GPU = core.use_gpu()
        model = models.Cellpose(gpu=use_GPU, model_type=self.model_type)
        self.mask, flows, styles, diams = model.eval(image_input, diameter=None, flow_threshold=None, channels=self.channels)
        self.table = pd.DataFrame(
            measure.regionprops_table(
                self.mask,
                intensity_image=image_input,
                properties=['area', 'label', 'centroid'])).set_index('label')
        return self.mask, self.table

    def stackObjects_cellpose_ebimage_parametrs_method(self, image_input ,extract_pixel, resize_pixel, img_label):
        '''
        fnction similar to the EBimage stackObjectsta, return a crop size based on center of measured mask
        :param extract_pixel: size of extraction acording to mask (e.g. 50 pixel)
        :param resize_pixel: resize for preforming tf prediction (e.g. 150 pixel)
        :param img_label: the mask value for stack
        :return: center image with out background
        '''
        img = image_input
        mask= self.mask
        table = self.table
        #table = table.astype({"centroid-0": 'int', "centroid-1": 'int'})
        x, y = table.loc[img_label, ["centroid-0", "centroid-1"]]
        x, y = int(x), int(y)
        mask_value = mask[x, y]
        x_start = x - extract_pixel
        x_end = x + extract_pixel
        y_start = y - extract_pixel
        y_end = y + extract_pixel
        if x_start < 0 or x_end < 0 or y_start < 0 or y_end < 0:
            stack_img = None
            mask_value = None
            return stack_img, mask_value
        else:
            mask_bin = np.zeros((np.shape(img)[0], np.shape(img)[1]), np.int32)
            mask_bin[mask == mask_value] = 1
            masked_image = img * mask_bin
            stack_img = masked_image[x_start:x_end, y_start:y_end]
            stack_img = skimage.transform.resize(stack_img, (resize_pixel, resize_pixel), anti_aliasing=False)
            return stack_img, mask_value

    def measure_properties(self, input_image):
        def sd_intensity(regionmask, intensity_image):
            return np.std(intensity_image[regionmask])

        def skew_intensity(regionmask, intensity_image):
            return skew(intensity_image[regionmask])

        def pixelcount(regionmask):
            return np.sum(regionmask)

        def mean_int(regionmask, intensity_image):
            return np.mean(intensity_image[regionmask])

        prop_names = [
            "label",
            "area",
            "eccentricity",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "inertia_tensor",
            "inertia_tensor_eigvals",
            "moments",
            "moments_central",
            "moments_hu",
            "moments_normalized",
            "orientation",
            "perimeter",
            "perimeter_crofton",
            # "slice",
            "solidity"
        ]
        table_prop = measure.regionprops_table(
            self.mask, intensity_image=input_image, properties=prop_names,
            extra_properties=(sd_intensity, skew_intensity, pixelcount, mean_int)
        )
        return table_prop

    def display_image_prediction(self,img ,prediction_table, font_select, font_size, windows=False, lable_draw = 'predict',round_n = 2):
        '''
        ch: 16 bit input image
        mask: mask for labale
        lable_draw: 'predict' or 'area'
        font_select: copy font to the working directory ("DejaVuSans.ttf" eg)
        font_size: 4 is nice size
        round_n: integer how many number after decimel

        return:
        info_table: table of objects measure
        PIL_image: 16 bit mask rgb of the labeled image
        '''
        # count number of objects in nuc['sort_mask']
        img_gs = img_as_ubyte(img)
        PIL_image = Image.fromarray(img_gs)
        # round
        info_table = prediction_table.round({'centroid-0': 0, 'centroid-1': 0})
        info_table['predict_round'] = info_table.loc[:, 'predict'].astype(float).round(round_n)
        info_table['area_round'] = info_table.loc[:, 'area'].astype(float).round(round_n)
        info_table = info_table.reset_index(drop=True)
        draw = ImageDraw.Draw(PIL_image)
        if lable_draw == 'predict':
            lable = 4
        else:
            lable = 5
        # use a bitmap font
        if windows:
            font = ImageFont.truetype("arial.ttf", font_size, encoding="unic")
        else:
            font = ImageFont.truetype(font_select, font_size)
        for i in range(len(info_table)):
            draw.text((info_table.iloc[i, 2].astype('int64'), info_table.iloc[i, 1].astype('int64')),
                      str(info_table.iloc[i, lable]), 'red', font=font)
        return info_table, PIL_image













