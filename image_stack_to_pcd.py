from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import pyvista as pv
import plotly.express as px
import matplotlib.pyplot as plt
#comment

def get_files_abs_paths (
		folder_path: str
		):
	""" """
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

	layer_arr = np.ones (thresholding_result.shape[0]) * layer_num

	layer = np.hstack (
			(
				thresholding_result.reshape(-1,2),
				layer_arr.reshape(-1,1)
				)
			)

	return layer


def process_files_in_folder (
		folder_path: str
		):

	files_path_list = get_files_abs_paths (folder_path = folder_path)

	print (f"IMAGE PROCESSING STAGE\n")

	stack_layers = {}
	for file_index, file_path in enumerate (files_path_list):

		print (f"[INFO] File [{file_index+1}/{len (files_path_list)}] is processing")

		temp_res = image_processing (file_path)
		temp_res = create_layer (
				thresholding_result = temp_res,
				layer_num = file_index
				)

		temp_name = os.path.basename (file_path).split ('/') [-1] [:-4]
		stack_layers [temp_name] = temp_res

	print('PROCESSING COMPLETED')

	return stack_layers


def tissue_reconstruction (
		stack: dict
		):

	temp_df = np.ones (3)
	print (f'The df size is: {temp_df.shape}')

	for layer_index, layer_name in enumerate(stack.keys()):

		print (f'Layer index is: [{layer_index+1}/{len (stack)}]')

		temp_df = np.vstack (
				(
					stack[layer_name],
					temp_df
					)
				)
		print (f'The df size is: {temp_df.shape}')

	return temp_df
