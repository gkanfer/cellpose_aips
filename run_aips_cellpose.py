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
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tfi
from utils import AIPS_cellpose as AC

if (__name__ == "__main__"):
    import argparse
    parser = argparse.ArgumentParser(description='AIPS activation')
    parser.add_argument('--file', dest='file', type=str, required=True,
                        help="The name of the image to analyze")
    parser.add_argument('--path', dest='path', type=str, required=True,
                        help="The path to the file")
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help="The name of the model")
    parser.add_argument('--path_model', dest='path_model', type=str, required=True,
                        help="The path to the model")
    parser.add_argument('--path_out', dest='path_out', type=str, required=True,
                        help="The path to saver binary files for upload")
    args = parser.parse_args()
    #upload model
    model_cnn  = load_model(os.path.join(args.path_model,args.model))
    AIPS_pose_object = AC.AIPS_cellpose(Image_name = args.file, path= args.path, model_type="cyto", channels=[0,0])
    img = AIPS_pose_object.cellpose_image_load()
    # create mask for the entire image
    mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img[0,:,:])

    table["predict"] = 'Na'
    test = []
    for i in range(len(table)):
        stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img[0,:,:],
                                                                                         extract_pixel=50, resize_pixel=150,
                                                                                         img_label=table.index.values[i])
        if stack is None:
            continue
        else:
            plt.imsave("temp.png", stack)
            test_imgs = img_to_array(load_img("temp.png", target_size=(150, 150)))
            test_imgs = np.array(test_imgs)
            # test_imgs_scaled = test_imgs.astype('float32')
            test_imgs_scaled = test_imgs
            test_imgs_scaled /= 255
            pred = model_cnn.predict(test_imgs_scaled.reshape(1, 150, 150, 3),verbose=0).tolist()[0][0]
            table.loc[i,"predict"] = pred
    # remove nas
    table_na_rmv = table.loc[table['predict']!='Na',:]
    # threshold for selected cells is 0.5
    # predict = np.where(table_na_rmv.loc[:,'predict'] > 0.5, 1, 0)
    # table_na_rmv = table_na_rmv.assign(predict = predict)
    # remove area smaller then 1500
    table_na_rmv = table_na_rmv.loc[table['area'] > 1500,:]
    ##### binary image contains the phnotype of intrse #####
    image_blank = np.zeros((np.shape(img[0,:,:])[0],np.shape(img[0,:,:])[1]))
    binary, table_sel = AIPS_pose_object.call_bin(table_sel_cor = table_na_rmv, threshold = 0.5 ,img_blank = image_blank)
    tfi.imsave(os.path.join(args.path_out, 'binary.tif'), binary)