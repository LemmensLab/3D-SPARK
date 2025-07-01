#Import packages
import sys
sys.path.insert(0, "..")  ## Change this to place where QuantSIM 'package' from GitHub repo is in your computer

import numpy as np
import pandas as pd
import re
import os
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import smbclient
import time
import shutil
import itertools
import bisect
from czifile import imread, CziFile
from skimage import filters, morphology
from skimage.transform import rescale
from skimage.measure import label,regionprops_table
from skimage.segmentation import find_boundaries
from sklearn.metrics import pairwise_distances
import tifffile as tif
import napari
from cellpose import models
import scipy.ndimage as scipy_image
import pyclesperanto_prototype as cle
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import scipy
import seaborn as sns
import xml.etree.ElementTree as ET
import warnings
from tqdm import tqdm
import gzip


## For visualizing a bunch of files upon starting Napari, should use keyword to show specific images and not all at once.
## layer identifier (001,102, etc.)
## Channel to focus on (everything that has channel_name AT THE END, so channel_name.extension)
## Analysis to focus on (everything that has channel1_vs_channel2)

def run_napari(image_results_folder, channel_names, channel_colors, keyword = ''):
    if not os.path.exists(image_results_folder):
        print('Path does not exist: {}'.format(image_results_folder))
        return None
    layer_folders = ['layers','overlap_layers']
    if len(channel_names) != len(channel_colors):
        raise Exception('Channel names and colors should have same number of elements.')
    viewer = napari.Viewer()
    for layer_folder in layer_folders:
        if not os.path.exists(os.path.join(image_results_folder,layer_folder)):
            continue
        layer_files = sorted(os.listdir(os.path.join(image_results_folder,layer_folder)))
        ## numpy arrays saved as compressed .gz files (.npy.gz)
        for file in layer_files:
            if file.endswith('.npy.gz') and keyword in file:
                f = gzip.GzipFile(os.path.join(image_results_folder,layer_folder,file), "r")
                layer_name = file[:-7]
                layer_data = np.load(f)
                if 'mask' in file:
                    for index,channel in enumerate(channel_names):
                        if layer_name.endswith(channel):
                            viewer.add_image(layer_data, name = layer_name, colormap = channel_colors[index], blending = 'additive')
                            break
                    else:
                        viewer.add_image(layer_data, name = layer_name, colormap = 'gray', blending = 'additive')
                elif 'segmented' in file:
                    viewer.add_labels(layer_data, name = layer_name)
                else:
                    for index,channel in enumerate(channel_names):
                        if layer_name.endswith(channel):
                            viewer.add_image(layer_data, name = layer_name, colormap = channel_colors[index], blending = 'additive')
                            break
    return viewer

def add_image_to_napari(napari_object, file_path, layer_name = '', color = 'gray'):
    f = gzip.GzipFile(file_path, "r")
    if layer_name == '':
        layer_name_start = len(file_path) - file_path[::-1].find('/')
        layer_name = file_path[layer_name_start:]
        layer_name = layer_name[:-7]
        layer_data = np.load(f)
    napari_object.add_image(layer_data, name = layer_name, colormap = color, blending = 'additive')

def add_labels_to_napari(napari_object, file_path, layer_name = ''):
    f = gzip.GzipFile(file_path, "r")
    if layer_name == '':
        layer_name_start = len(file_path) - file_path[::-1].find('/')
        layer_name = file_path[layer_name_start:]
        layer_name = layer_name[:-7]
        layer_data = np.load(f)
    napari_object.add_labels(layer_data, name = layer_name)