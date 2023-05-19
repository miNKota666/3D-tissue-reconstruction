from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import pyvista as pv
import plotly.express as px
import matplotlib.pyplot as plt


# comment

def get_files_abs_paths (
        folder_path: str
        ):
    """
    From the certain folder create a string-format object (file.names), that includes a list of 'n' files in the
    folder. Then, a subsequent extraction of the short path (key endings) of a file to the temp_list for further
    processing or storage is accomplished.
    :param: function receives a folder_path in a string format.
    :return: function returns the temp_list in a string format. The intended purpose of the returned object is a file's
    path in the folder in a cut format
    """
    print (f"Opened folder:\n{folder_path}")
    file_names = os.listdir (folder_path)
    print (f"There are {len (file_names)} files in the folder\n")

    temp_list = []
    for file_name in file_names:
        temp_list.append (os.path.abspath (f"{folder_path}/{file_name}"))

    return temp_list


def image_processing (
        image_path: str
        ):
    '''
    The function takes an image by the specified path (image.path). The received image is blurred (Gaussian blur)
    with standard deviation set at 9 for both X and Y directions. Then, the image proceeds to the direct Otsu's
    thresholding (cv.THRESH_BINARY+cv.THRESH_OTSU) with a subsequent extraction of non-zero values to the thrshInv
    np.array object.
    :param: function receives an image_path in a string format.
    :return: function returns the NumPy object (np.array) threshInv. The intended purpose of the returned object is a
    set of non-zero values (highlighted features) in the studied image.
    '''
    # read img in gray format
    img = cv2.imread (image_path, 0)

    # blur image for further processing and Otsu's thresholding
    blurred = cv2.GaussianBlur (img, (9, 9), 0)

    # Otsu's thresholding
    (T, threshInv) = cv2.threshold (
            src = blurred,
            thresh = 0,
            maxval = 255,
            type = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
    print ("[INFO] Otsu's thresholding value: {}".format (T))

    threshInv = np.argwhere (threshInv == 0)
    print ("[INFO] Otsu's thresholding completed\n")

    return threshInv


def create_layer (
        thresholding_result: np.array,
        layer_num: int
        ):
    '''
    The function creates a layer array by combining the thresholding results with the corresponding layer number.
    Thereby, each element in the layer_arr corresponds to the given layer_num. Next, the function uses np.hstack to
    horizontally stack the reshaped thresholding_result array (reshaped into a 2D array with two columns) with the
    reshaped layer_arr array (reshaped into a 2D array with one column). This combines the thresholding results and
    the layer numbers into a single array, where each row represents a data point with two thresholding values and a
    corresponding layer number.
    :param: function receives the thresholding_result NumPy array.
    :param: function receives the layer_num integer.
    :return: function returns the NumPy object (np.array) layer. The intended purpose of the returned object is a
    set of reshaped layers for the further processing.
    '''
    layer_arr = np.ones (thresholding_result.shape [0]) * layer_num

    layer = np.hstack (
            (
                thresholding_result.reshape (-1, 2),
                layer_arr.reshape (-1, 1)
                )
            )

    return layer


def process_files_in_folder (
        folder_path: str
        ):
    '''
    The function calls the get_files_abs_paths function to get the absolute paths for the files in the folder. Then,
    for each file in the directory it performs image processing by image_processing function (stored in temp_res),
    creates a layer array from the thresholding results (stored in temp_res), removes the file extension from the
    file name, and step-by-step adds the temp_res layer array to the stack_layers dictionary using the extracted file
    name as the key.
    :param: function receives the folder_path string.
    :return: function returns the NumPy object (dictionary) stack_layers. The intended purpose of the returned
    object is the set of layer arrays ready for the combining.
    '''
    files_path_list = get_files_abs_paths (folder_path = folder_path)

    print (f"IMAGE PROCESSING STAGE\n")

    stack_layers = {}
    for file_index, file_path in enumerate (files_path_list):
        print (f"[INFO] File [{file_index + 1}/{len (files_path_list)}] is processing")

        temp_res = image_processing (file_path)
        temp_res = create_layer (
                thresholding_result = temp_res,
                layer_num = file_index
                )

        temp_name = os.path.basename (file_path).split ('/') [-1] [:-4]
        stack_layers [temp_name] = temp_res

    print ('PROCESSING COMPLETED')

    return stack_layers


def tissue_reconstruction (
        stack: dict
        ):
    '''
    The function firstly creates a 3-dimensional array filled with ones. Subsequently, the function
    performs the vertically stacking of the processed layer array on the top of the temp_df NumPy array. This step is
    then performed for each layer kept in the stack_layers dictionary.
    :param: function receives the stack NumPy dictionary.
    :return: function returns the NumPy object (3-dimensional array) temp_df. The intended purpose of the returned
    object is the combined layer for the subsequent processing.
    '''
    temp_df = np.ones (3)
    print (f'The df size is: {temp_df.shape}')

    for layer_index, layer_name in enumerate (stack.keys ()):
        print (f'Layer index is: [{layer_index + 1}/{len (stack)}]')

        temp_df = np.vstack (
                (
                    stack [layer_name],
                    temp_df
                    )
                )
        print (f'The df size is: {temp_df.shape}')

    return temp_df
