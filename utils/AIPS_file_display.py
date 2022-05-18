import tifffile as tfi
import numpy as np
from skimage.filters import threshold_local
from scipy.ndimage.morphology import binary_opening,binary_erosion
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from utils.display_and_xml import evaluate_image_output,test_image

class Compsite_display(object):
    def __init__(self,input_image, mask_roi):
        self.input_image = input_image
        self.mask_roi = mask_roi

    def outline_seg(mask,index):
        '''
        outlined single ROI
        mask: binary string, "no-mask" or "mask" (inherit mask from __init__)
        index = ROI value
        :return
        '''
        seg_mask_temp = np.zeros(np.shape(mask), dtype=np.int32)
        seg_mask_temp[mask == index] = index
        seg_mask_eros_9 = binary_erosion(seg_mask_temp, structure=np.ones((9, 9))).astype(np.float64)
        seg_mask_eros_3 = binary_erosion(seg_mask_temp, structure=np.ones((3, 3))).astype(np.float64)
        framed_mask = seg_mask_eros_3 - seg_mask_eros_9
        return framed_mask

    def binary_frame_mask(ch,mask, table=None):
        '''
        Create a mask for NIS-elements to photo-activate for multiple point
        :parameter
        input_image_str: binary string,input image is-  "seed" or "target" (inherit from self)
        mask_str - input mask (RGB) 32integer
        :return
        framed_mask (RGB)
        '''
        if table is None:
            info_table = pd.DataFrame(
                measure.regionprops_table(
                    mask,
                    intensity_image=ch,
                    properties=['area', 'label', 'centroid'],
                )).set_index('label')
        else:
            info_table = table
        info_table['label'] = range(2, len(info_table) + 2)
        framed_mask = np.zeros(np.shape(mask), dtype=np.int32)
        if len(info_table['label']) > 1:
            for i in info_table.index.values:
                framed_mask = framed_mask + Compsite_display.outline_seg(mask, i)
        else:
            framed_mask = Compsite_display.outline_seg(mask, info_table.index.values)
        return framed_mask

    def draw_ROI_contour(self, channel=None):
        ''' create a 3 channel rgb image from 16bit input image
            optional bin contour image from ROI image
            :parameter
            input_image: binary string,input image is-  "seed" or "target"
            mask: binary string, "no-mask" or "mask" (inherit mask from __init__)
            channel: 0,1,2 (rgb)
            :return
            3 channel stack file 8bit image with a
        '''
        input_gs_image = self.input_image
        input_gs_image = (input_gs_image / input_gs_image.max()) * 255
        ch2_u8 = np.uint8(input_gs_image)
        rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
        rgb_input_img[:, :, 0] = ch2_u8
        rgb_input_img[:, :, 1] = ch2_u8
        rgb_input_img[:, :, 2] = ch2_u8
        mask_input = self.mask_roi
        if len(np.unique(mask_input)) > 1:
            bin_mask = Compsite_display.binary_frame_mask(ch=self.input_image,mask=self.mask_roi)
            bin_mask = np.where(bin_mask == 1, True, False)
            if channel is not None:
                rgb_input_img[bin_mask > 0, channel] = 255
            else:
                rgb_input_img[bin_mask > 0, 2] = 255
        return rgb_input_img

    def measure_properties(self):
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
            self.mask_roi, intensity_image=self.input_image, properties=prop_names
        )
        return table_prop




