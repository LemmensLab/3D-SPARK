#Import packages
import sys
sys.path.insert(0, "..")  ## Change this to place where QuantSIM 'package' from GitHub repo is in your computer

import numpy as np
import pandas as pd
import os
import re
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import time
import shutil
from czifile import imread, CziFile
import cv2
import tifffile as tiff
from skimage import filters, morphology,segmentation
from skimage.transform import rescale
from skimage.measure import label,regionprops_table
from sklearn.metrics import pairwise_distances
import pyclesperanto_prototype as cle
import napari_segment_blobs_and_things_with_membranes as nsbatwm
import xml.etree.ElementTree as ET
import warnings
import gzip
import raster_geometry as rg
from QuantSIM.common import check_to_make_dir, find_key_from_pattern, update_labels_maxima

def slide_channel_threshold(SLIDE_PATH, image_suffixes = '_Out', channel_names = ['EDU','dUTP'], method = 'otsu'):
    '''
    Calculate intensity threshold for each channel based on all images from the slide.
    Assumption: all images taken from the same slide were performed with same microscope settings (laser power, gain, reconstruction...).
    DOES NOT WORK WELL, each cell has different signal uptake, which affects the intensity of the SIM artifacts in an image-to-image basis, so the threshold should be calculated per image.
    '''
    stitch_maxproj_channels = []
    for channel in channel_names:
        stitch_maxproj_channels.append(np.empty((0,0)))
    channels_wavelengths_raw = []
    if '.' in image_suffixes:
        image_suffixes = image_suffixes[0:image_suffixes.find('.')]
    for file in os.listdir(SLIDE_PATH):
        file_match = re.match(r'Image \d+({})\.czi$'.format(image_suffixes), file)
        if file_match != None:
            if file_match.group(1) != '':
                IMAGE_PATH = os.path.join(SLIDE_PATH, file)
            else:
                continue
        else:
            continue
        channels_image = []
        metadata = CziFile(IMAGE_PATH)
        xml_metadata = ET.ElementTree(ET.fromstring(metadata.metadata()))
        for element in xml_metadata.iter('ExcitationWavelength'):
            channels_image.append(element.text)
        channels_wavelengths_raw.append(channels_image)
        stack = np.squeeze(imread(IMAGE_PATH))
        if stack.shape[0] != len(channel_names):
            raise Exception('Number of channels does not match number of channel names given.')
        for channel in range(stack.shape[0]):
            if stitch_maxproj_channels[channel].shape == (0,0):
                stitch_maxproj_channels[channel] = np.array(cle.maximum_z_projection(np.asarray(stack[channel,:,:,:])))
            else:
                stitch_maxproj_channels[channel] = np.concatenate((stitch_maxproj_channels[channel],
                                                                np.array(cle.maximum_z_projection(np.asarray(stack[channel,:,:,:])))),
                                                                axis = 0)
    if not all(i==channels_wavelengths_raw[0] for i in channels_wavelengths_raw):
        for i in channels_wavelengths_raw:
            print(i)
        raise Exception('Not all images have same channel order. Issue not handled yet.')
    int_thresh = []
    if method == 'otsu':
        for channel in range(len(stitch_maxproj_channels)):
            int_thresh.append(filters.threshold_otsu(stitch_maxproj_channels[channel]))
    return(int_thresh)

def moments_thresholding(img):
    '''
    Image thresholding method using the moment-preserving principle.

    Little knowledge in what this means, but this threshold is the most consistent in segmenting regions in max-z projections regardless of density of signal per nucleus.

    Original method described by Wen-Hsiang Tsai, 1985 (https://doi.org/10.1016/0734-189X(85)90133-1)
    Full credits to PaPe answer in StackOverlfow (https://stackoverflow.com/questions/53738798/transfer-auto-threshold-moments-method-from-image-j-to-python)
    '''
    #Instead of skimage.exposure.histogram (img, nbins=256)   
    grey_value = np.arange(256)
    img_pixel_freq = cv2.calcHist([img],[0],None,[256],[0,255])
    img_pixel_freq = img_pixel_freq.reshape((256,))
    img_pixel_freq = np.int64(img_pixel_freq)
    img_pixel_hist = (img_pixel_freq, grey_value)
    pix_sum = img.shape[0]*img.shape[1] # calculating the number of pixels

    #from the paper, calculating the 3 first orders

    pj = img_pixel_hist[0] / pix_sum 
    pj_z1 = np.power(img_pixel_hist[1], 1) * pj
    pj_z2 = np.power(img_pixel_hist[1], 2) * pj
    pj_z3 = np.power(img_pixel_hist[1], 3) * pj

    m0 = np.sum(pj)
    m1 = np.sum(pj_z1)
    m2 = np.sum(pj_z2)
    m3 = np.sum(pj_z3)
    print(m0,m1,m2,m3)
    cd = (m0*m2) - (m1*m1)
    c0 = ((-m2*m2) - (-m3*m1))/cd
    c1 = ((m0*-m3) - (m1*-m2))/cd


    z0 = 0.5 *(-c1 - (np.power(np.power(c1, 2) - 4*c0, 1/2)))
    z1 = 0.5 *(-c1 + (np.power(np.power(c1, 2) - 4*c0, 1/2)))

    pd = z1 - z0
    p0 = (z1 - m1) / pd # p0 should be the percentage of the pixels to which the threshold t should be done

    # using cumulative histogram and comparing it to a target value by calculating the difference. When the difference is the lowest, the index indicates the value of the threshold t
    cum_pix = np.cumsum(img_pixel_freq) 
    target_value = p0 * pix_sum

    diff = [(i - target_value) for i in cum_pix]

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    t = diff.index(find_nearest(diff, 0))

    return t

class SIM_segmentation:
    '''
    Python class to perform image analysis in a single stacked SIM image. Based on Robert Welch original script.
    
    Identifier: 000

    Arguments:
        STACK_PATH: string. Absolute or relative path indicating the image stack.
        RES_IMAGE_PATH: string. Absolute or relative, existing or non-existing path indicating folder to store results.
        DAPI_PATH: string. Absolute or relative path to independent DAPI channel, for automated nuclei segmentation (will turn into 2D).
        MASK_PATH: string. Absolute or relative path to tiff file containing mask of region of interest in XY.
        channel_names: list of strings. Names of channels for following analysis, will refer internally to each channel by this assigned names from now on.
        channel_colors: list of strings. Names of colors compatible with Napari for visualization (magenta, cyan, yellow, green, red, blue, white).
        channel_org: list of strings. Names of channels in the order that appear in the provided image. There may be more channels than in "channel_names", which will be skipped. Used for reordering or skipping channels in downstream analysis.
        override: boolean. If True, will erase all files in the analysis folder (BE CAREFUL). The mentioned analysis folder is 'RES_IMAGE_PATH/auto_analysis'
    '''
    def __init__(self, STACK_PATH, RES_IMAGE_PATH, DAPI_PATH = None, MASK_PATH = None, channel_names = ['EDU','UTP'], channel_colors = ['magenta','cyan'], channel_org = ['EdU','UTP'], override = False):
        RES_IMAGE_PATH = os.path.relpath(RES_IMAGE_PATH)
        if 'auto_analysis' not in RES_IMAGE_PATH.lower() and not RES_IMAGE_PATH.lower().endswith('auto_analysis'):
            RES_IMAGE_PATH = RES_IMAGE_PATH + '/auto_analysis'
        print('--- Output folder: ' + RES_IMAGE_PATH)
        if not os.path.exists(RES_IMAGE_PATH):
            os.mkdir(RES_IMAGE_PATH)
        if override == True:
            shutil.rmtree(RES_IMAGE_PATH)
        if not os.path.exists(RES_IMAGE_PATH):
            os.mkdir(RES_IMAGE_PATH)
        if not os.path.exists(STACK_PATH):
            raise Exception('Image stack could not be found.')
        self.STACK_PATH = STACK_PATH
        self.RES_IMAGE_PATH = RES_IMAGE_PATH
        self.ROOT_DIR = str(Path(STACK_PATH).parent)
        self.IMAGE_NAME = STACK_PATH.split('/')[-1].strip('.czi')
        self.DAPI_PATH = DAPI_PATH
        self.MASK_PATH = MASK_PATH
        self.override = override
        allowed_channel_colors = ['red','green','blue','yellow','magenta','cyan','white']
        for color in channel_colors:
            if color not in allowed_channel_colors:
                raise Exception('Channel colors selected not allowed. Have to choose between: {}'.format(str(allowed_channel_colors)))
        if len(channel_names) != len(channel_colors):
            channel_names = channel_names[0:min([len(channel_names),len(channel_colors)])]
            channel_colors = channel_colors[0:min([len(channel_names),len(channel_colors)])]
        self.channel_names = channel_names
        self.channel_colors = dict(zip(channel_names,channel_colors))
        self.info_vector = {}
        self.time_log = {}
        self.time_log['time-0'] = time.time()
        self.start_time = time.time()
        # # # # # # # # PRE-PROCESSING # # # # # # #
        #Convert 6D (SCTZYX) image information to 4D numpy array [channels,focal_plane,Y_dim,X_dim]
        stack = np.squeeze(imread(STACK_PATH))
        self.layers = {}
        self.tables = {}
        self.layers['000_channels'] = {}
        for count,channel in enumerate(channel_org):

            if channel not in channel_names:
                continue #Skip channel if not part of channels to be analyzed.
            if len(stack.shape) == 4:
                self.layers['000_channels'][channel] = np.asarray(stack[count,:,:,:])
            else:
                self.layers['000_channels'][channel] = np.asarray([stack[count,:,:]])
            if count == 0:
                self.ori_dims = self.layers['000_channels'][channel].shape
        if len(list(self.layers['000_channels'].keys())) != len(channel_names):
            print('WARNING: NUMBER OF CHANNELS INDICATED HIGHER THAN NUMBER OF CHANNELS IN IMAGE')
            print('---CHANNELS INDICATED: '+str(channel_org))
            print('---CHANNELS TO EXTRACT: '+str(channel_names))
            del self.layers
            del self.tables # Didn't think a better way to stop moving forward than forcing self-deletion.
        if DAPI_PATH != None:
            DAPI_frame = np.squeeze(imread(DAPI_PATH))
            ## This could be a processed DAPI frame (Y,X), or processed DAPI stack (Z,Y,X), or unprocessed DAPI (angle,phase,Y,X)
            ## To get simplified results, turn into YX by maxproj, use that as reference of where cells are.
            DAPI_dims = DAPI_frame.shape
            if len(DAPI_dims) == 2:
                DAPI_frame = np.expand_dims(DAPI_frame, axis = 0)
            elif len(DAPI_dims) > 3:
                DAPI_frame = DAPI_frame.reshape(np.prod(np.array(DAPI_dims[0:len(DAPI_dims)-2])),DAPI_dims[-2],DAPI_dims[-1])
            DAPI_maxproj = np.array(cle.maximum_z_projection(DAPI_frame))
            ## It is possible the dimensions of DAPI and actual signal image stacks do not match. Processing increases resolution depth close to 2x in the YX dimensions. Have to scale DAPI to actual image dimensions.
            DAPI_maxproj = rescale(DAPI_maxproj, [stack.shape[len(stack.shape)-2]/DAPI_maxproj.shape[0], stack.shape[len(stack.shape)-1]/DAPI_maxproj.shape[1]])
            DAPI_maxproj = filters.gaussian(DAPI_maxproj, sigma = 1)
            # Identify bright DAPI that is big enough to be an individual nucleus, even if they are close together.
            DAPI_mask_stringent = DAPI_maxproj > filters.threshold_otsu(DAPI_maxproj)
            DAPI_mask_stringent[morphology.remove_small_objects(label(DAPI_mask_stringent), 10000) == 0] = 0 
            # Identify dim DAPI, recognizing all potential signal that could be deemed positive for DAPI.
            #DAPI_mask_loose = DAPI_maxproj > filters.threshold_multiotsu(DAPI_maxproj)[0]
            #DAPI_mask_loose[morphology.remove_small_objects(label(DAPI_mask_loose), 10000) == 0] = 0
            # Using bright DAPI as seeds of true nuclei, extend to the limits of dim DAPI to identify borders of each cell.
            DAPI_labeled_expanded_start = label(DAPI_mask_stringent)
            #DAPI_labeled_expanded = (segmentation.expand_labels(DAPI_labeled_expanded_start, 50)*DAPI_mask_loose).astype(int)
            #while np.sum(DAPI_labeled_expanded > 0) != np.sum(DAPI_labeled_expanded_start > 0):
            #    DAPI_labeled_expanded_start = copy.copy(DAPI_labeled_expanded)
            #    DAPI_labeled_expanded = (segmentation.expand_labels(DAPI_labeled_expanded, 50)*DAPI_mask_loose).astype(int)
            DAPI_labeled_expanded = self.binary_morphology_fast(DAPI_labeled_expanded_start, "closing", circle = True)
            DAPI_labeled_expanded = segmentation.expand_labels(DAPI_labeled_expanded, 10) # One final extension, to fill gaps and get contour a bit more broad.
            DAPI_labeled_limits = segmentation.find_boundaries(DAPI_labeled_expanded)
            DAPI_mask = DAPI_labeled_expanded.astype(bool).astype(int)*np.logical_not(DAPI_labeled_limits) # Make binary mask but leaving space between adjacent nuclei.
            #DAPI_mask = self.binary_morphology_fast(DAPI_mask,mode='dilation',morph_window_length=120, circle = True)
            self.layers['000_DAPI_ref'] = DAPI_maxproj
            self.layers['000_DAPI_mask'] = DAPI_mask
        if MASK_PATH != None:
            MASK = np.squeeze(tiff.imread(MASK_PATH))
            ## This could be a frame (Y,X) or a stack (Z,Y,X). The important thing is that it has to be a binary file (1 is mask, 0 is irrelevant space).
            if(np.sum(MASK > 1)) > 0:
                MASK[np.where(MASK > 1)] = 1
                MASK[np.where(MASK < 1)] = 0
            if len(MASK.shape) == 2:    # Can't remember why this was relevant in DAPI. It is a distinction between 3D or 2D, but there is no need to expand dimensions yet? Will keep just in case, future 3D masks.
                MASK = np.expand_dims(MASK, axis = 0)
            MASK_maxproj = np.array(cle.maximum_z_projection(MASK))
            self.layers['000_MASK_ref'] = MASK_maxproj
            self.layers['000_MASK_mask'] = MASK
            
    def binary_morphology_fast(self, labeled_img, mode, morph_window_length = 120, circle = False):
        '''
        minimum window: 5 for square, 12 for circle.
        '''
        ## To make sure there is enough room to extend the shape and does not halt due to borders, will add a frame as big as the closing window on each side.
        labeled_img_extend = np.zeros(np.array(labeled_img.shape)+morph_window_length*2)
        labeled_img_extend[morph_window_length:(labeled_img.shape[0]+morph_window_length),
                           morph_window_length:(labeled_img.shape[1]+morph_window_length)] = copy.copy(labeled_img)
        if mode not in ['dilation','erosion','opening','closing']:
            raise Exception('Invalid mode for binary morphological transformation. Only allowed keywords: dilation, erosion, opening, closing.')
        if circle:
            n_iterations = morph_window_length//12
            footprint = rg.circle(12,6).astype(int) #shape of array,radius
        else:
            n_iterations = morph_window_length//5
            footprint = np.ones((5,5))
        ## Since it is a computationally heavy process, chopped in stages.
        if mode == 'dilation' or mode == 'closing':
            for i in range(n_iterations):
                labeled_img_extend = morphology.binary_dilation(labeled_img_extend, footprint = footprint)
        if mode != 'dilation':
            for i in range(n_iterations):
                labeled_img_extend = morphology.binary_erosion(labeled_img_extend, footprint = footprint)
        if mode == 'opening':
            for i in range(n_iterations):
                labeled_img_extend = morphology.binary_dilation(labeled_img_extend, footprint = footprint)
        labeled_img_extend = labeled_img_extend[morph_window_length:(labeled_img.shape[0]+morph_window_length),
                                                morph_window_length:(labeled_img.shape[1]+morph_window_length)]
        ## Compare to initial mask, in case something weird happens with signal close to image edges. OR logic.
        labeled_img_closed = np.array(labeled_img + labeled_img_extend, dtype=bool).astype(int)
        return(labeled_img_closed)

    def image_truncation(self, auto_mode = True, trunc_channel = None, manual_center = None, manual_YX_dims = (300,300),
                         threshold_method = 'otsu', inherit_threshold = None, circle_closing = False):
        '''
        Find region of image containing nuclei information and crop. Reduces size of array, more efficient downstream.
        If there is a DAPI layer (DAPI_PATH in SIM_segmentation object calling), will automatically use it to identify cells.
            Selection of relevant cells will depend on trunc_channel in this case.

        Identifier: 001

        Modifies: 000_channels

        auto_mode: logical. If True, locate the cell based on the combined signal of all channels. If False, truncate center of image.
        trunc_channel: list of strings matching channel names in Segmentation object. If None, will use all channel names in object.
        manual_center: list of two integers. Default None, applicable when auto_mode == False. YX location of central pixel of the rectangle to crop. If None, will use center of image.
        manual_YX_dims: tuple of two integers. Default (300,300), applicable when auto_mode == False. Length of rectangle sides to crop in YX pixels, centered by manual_center.
        threshold_method: string. Default 'otsu'. Intensity-based thresholding method used to generate max Z projection masks of signal and locate nuclei.
            If DAPI channel is available (self.DAPI_path != None), will use DAPI mask directly, and intensity-based thresholding is used to evaluate which nuclei have signal for segmentation.
        inherit_threshold: list of floats. Default None. Intensity thresholds to define positive signal masks in every channel, overriding intensity-based thresholding calculations.
            Has to be as long as the number of channels in the stack (not only the ones specified in trunc_channels). If None, ignored.
        '''
        if self.DAPI_PATH != None:
            DAPI_labeled = label(self.layers['000_DAPI_mask'])
            DAPI_nuclei_n = np.max(DAPI_labeled)
            if DAPI_nuclei_n == 0:
                raise Exception('no nuclei without touching image borders identified with DAPI image.')
            ## If more than one nuceli based on DAPI, will decide which one to take based on trunc_channel signals.
        if trunc_channel == None:
            trunc_channel = copy.copy(self.channel_names)
        if type(trunc_channel) != list:
            trunc_channel = [trunc_channel]
        if inherit_threshold != None:
                if type(inherit_threshold) != list:
                    inherit_threshold = [inherit_threshold]
        for channel in trunc_channel:
            if channel not in self.channel_names:
                raise Exception('{} channel does not exist in the SIM analysis object. Available channels: {}'.format(trunc_channel,str(self.channel_names)))
        channels_maxzproj = {}
        channels_maxzproj_threshold = {}
        channels_maxzproj_roi_mask = {}
        channels_maxzproj_roi_mask_closing = {}
        for channel in self.channel_names:
            channels_maxzproj[channel] = np.array(cle.maximum_z_projection(self.layers['000_channels'][channel]))
            if inherit_threshold != None:
                channels_maxzproj_threshold[channel] = inherit_threshold[self.channel_names.index(channel)]
            elif threshold_method == 'otsu':
                channels_maxzproj_threshold[channel] = filters.threshold_otsu(channels_maxzproj[channel])
            elif threshold_method == 'moments':
                channels_maxzproj_threshold[channel] = moments_thresholding(channels_maxzproj[channel]*255)/255
            channels_maxzproj_roi_mask[channel] = channels_maxzproj[channel] > channels_maxzproj_threshold[channel]
            ## Apply a closing to fill holes or fuse close enough blobs into single region
            channels_maxzproj_roi_mask_closing[channel] = self.binary_morphology_fast(channels_maxzproj_roi_mask[channel],mode='closing',circle=circle_closing)
        if self.MASK_PATH != None:  ## Assumes mask is a binary, the mask contains a single nucleus.
            roi_mask_combined = copy.copy(self.layers['000_MASK_ref'])
            roi_mask_single_cell = label(roi_mask_combined)
            max_area_row = pd.DataFrame(regionprops_table(roi_mask_single_cell, properties = ('label','area','bbox'))).iloc[0,:]
        elif self.DAPI_PATH != None:  ## Locates cells based on DAPI (DAPI turned into a segmented and labeled version when calling the object for the first time).
            roi_mask_combined = copy.copy(self.layers['000_DAPI_mask'])
            if DAPI_nuclei_n > 1:
                DAPI_labeled_plus_channels_signal = copy.copy(DAPI_labeled)
                ## If more than one valid nuclei, choose the one with most signal from every channel used for truncation.
                for channel in trunc_channel:
                    DAPI_labeled_plus_channels_signal = DAPI_labeled_plus_channels_signal*channels_maxzproj_roi_mask_closing[channel]
                DAPI_pixel_count = []
                for pixel_label in range(1,DAPI_nuclei_n+1):
                    DAPI_pixel_count.append(np.sum(DAPI_labeled_plus_channels_signal == pixel_label))
                DAPI_max_signal_label = DAPI_pixel_count.index(max(DAPI_pixel_count))+1
                roi_mask_single_cell = label(DAPI_labeled == DAPI_max_signal_label)
            else:
                roi_mask_single_cell = label(DAPI_labeled)
            max_area_row = pd.DataFrame(regionprops_table(roi_mask_single_cell, properties = ('label','area','bbox'))).iloc[0,:]
        else:
            roi_mask_combined = np.ones(channels_maxzproj[channel].shape)
            for channel in trunc_channel:
                roi_mask_combined = roi_mask_combined*channels_maxzproj_roi_mask_closing[channel]
            roi_mask_combined_labeled = label(roi_mask_combined)
            roi_mask_combined_props = pd.DataFrame(regionprops_table(roi_mask_combined_labeled, properties = ('label','area','bbox')))
            roi_mask_areas = roi_mask_combined_props.loc[:,'area'].to_numpy().flatten().tolist()
            max_area = max(roi_mask_areas)
            max_area_idx = roi_mask_areas.index(max_area)
            max_area_row = roi_mask_combined_props.iloc[max_area_idx,:]
            if max_area_row['area'] != max_area:
                raise Exception('Problem inside function "image_truncation". Max area entry not retrieved properly')
            max_area_label = max_area_row['label']
            ## Isolate largest label
            roi_mask_single_cell = np.zeros(channels_maxzproj[channel].shape)
            roi_mask_single_cell[roi_mask_combined_labeled == max_area_label] = 1
            ## Repeat closing in isolated label, so it covers as much of the nucleus as possible.
            roi_mask_single_cell = self.binary_morphology_fast(roi_mask_single_cell, mode = 'closing', morph_window_length=500,circle=circle_closing)
        max_area_bbox = max_area_row.loc[['bbox-0','bbox-1','bbox-2','bbox-3']].to_numpy().flatten().tolist()
        if auto_mode:
            lower_y = max([0,max_area_bbox[0] - 100])
            lower_x = max([0,max_area_bbox[1] - 100])
            higher_y = min([channels_maxzproj[channel].shape[0], max_area_bbox[2] + 100])
            higher_x = min([channels_maxzproj[channel].shape[1], max_area_bbox[3] + 100])
        else:
            if manual_center == None:
                manual_center = np.round(channels_maxzproj[channel].shape/2,0).astype(int)
            manual_YX_dims = np.array(manual_YX_dims)
            manual_YX_dims_halved = np.round(manual_YX_dims/2,0).astype(int)
            lower_y = max([0,manual_center[0] - manual_YX_dims_halved[0]])
            lower_x = max([0,manual_center[1] - manual_YX_dims_halved[1]])
            higher_y = min([channels_maxzproj[channel].shape[0], manual_center[0] + manual_YX_dims_halved[0]])
            higher_x = min([channels_maxzproj[channel].shape[1], manual_center[1] + manual_YX_dims_halved[1]])
        lower_y = int(lower_y)
        lower_x = int(lower_x)
        higher_y = int(higher_y)
        higher_x = int(higher_x)
        self.layers['001_signal_mask'] = roi_mask_combined[lower_y:higher_y,lower_x:higher_x]
        self.layers['001_roi_mask'] = roi_mask_single_cell[lower_y:higher_y,lower_x:higher_x]

        self.trunc_channels = trunc_channel
        self.channels_threshold_trunc = channels_maxzproj_threshold
        self.trunc_refcoord = [lower_y,lower_x]
        for channel in self.channel_names:
            self.layers['000_channels'][channel] = self.layers['000_channels'][channel][:,lower_y:higher_y,lower_x:higher_x]
        self.trunc_dims = self.layers['000_channels'][self.channel_names[0]].shape
        
        plot_channels = self.channel_names
        if self.DAPI_PATH != None:
            plot_channels = ['DAPI'] + plot_channels
        fig,ax = plt.subplots(5,max(2,len(plot_channels)))
        fig.set_figheight(5*2.5)
        fig.set_figwidth(max(2,len(plot_channels))*4)
        for count,channel in enumerate(plot_channels):
            if channel == 'DAPI':
                ax[0,count].imshow(self.layers['000_DAPI_ref'])
                ax[1,count].imshow(self.layers['000_DAPI_mask'], cmap='Greys')
                ax[2,count].imshow(self.layers['000_DAPI_ref'][lower_y:higher_y,lower_x:higher_x])
                ax[3,count].imshow(self.layers['000_DAPI_ref'][lower_y:higher_y,lower_x:higher_x])
                ax[3,count].imshow(self.layers['001_signal_mask'], cmap='Greys')
                ax[4,count].imshow(self.layers['000_DAPI_ref'][lower_y:higher_y,lower_x:higher_x])
                ax[4,count].imshow(self.layers['001_roi_mask'], 'Greys', alpha = 0.4)
            else:
                ax[0,count].imshow(channels_maxzproj[channel])
                ax[1,count].imshow(channels_maxzproj_roi_mask_closing[channel], cmap='Greys')
                ax[2,count].imshow(channels_maxzproj[channel][lower_y:higher_y,lower_x:higher_x])
                ax[3,count].imshow(channels_maxzproj[channel][lower_y:higher_y,lower_x:higher_x])
                ax[3,count].imshow(self.layers['001_signal_mask'], cmap='Greys')
                ax[4,count].imshow(channels_maxzproj[channel][lower_y:higher_y,lower_x:higher_x])
                ax[4,count].imshow(self.layers['001_roi_mask'], 'Greys', alpha = 0.4)
            ax[0,count].set_title(channel +' Original', fontsize = 8)
            ax[1,count].set_title(channel +' Signal Detection & Closing', fontsize = 8)
            ax[2,count].set_title(channel +' Truncated', fontsize = 8)
            ax[3,count].set_title(channel +' Truncated with Signal Detection', fontsize = 8)
            ax[4,count].set_title('Final ROI Mask', fontsize = 8)
        fig.suptitle('Image truncation', fontsize = 10)
        plt.tight_layout()
        plt.savefig(self.RES_IMAGE_PATH+'/image_truncation.png')
        plt.close()
        
        self.time_log['Image_truncation'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, image truncation: ' + self.time_log['Image_truncation'])
        self.start_time = time.time()
        
    def anisotropy_correction(self):
        '''
        Check for image isotropy (equal scaling between across 3 dimensions) and correct for it.

        Identifier: 002
        Modifes: 000_channels
        '''
        metadata = CziFile(self.STACK_PATH)
        xml_metadata = ET.ElementTree(ET.fromstring(metadata.metadata()))
        root = xml_metadata.getroot()

        voxelsizes = {'ScalingZ':0,'ScalingY':0,'ScalingX':0}
        for scaling in voxelsizes.keys():
            scale_val = [elem.text for elem in root.iter(scaling)][0]
            ## voxelsizes are in meters, they are very small so we have to multiply to get workable scaling factors for downstream work.
            scale_val = round(float(scale_val)*10**9,2)
            voxelsizes[scaling] = scale_val

        # Identify minimum scaling value (most of the time, X and Y have same scaling, and Z has increased scaling i.e. more space between slices)
        min_scale = min(voxelsizes.values())
        
        # upscaling performed in skimage uses as variables the image and the scaling for every dimension in ZYX. Since the dictionary stores in that order, can directly apply.
        for channel in self.channel_names:
            self.layers['000_channels'][channel] = rescale(self.layers['000_channels'][channel], [i/min_scale for i in list(voxelsizes.values())])
        self.voxelsizes = voxelsizes
        self.time_log['Anisotropy_correction'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, anisotropy correction: ' + self.time_log['Anisotropy_correction'])
        self.start_time = time.time()

    def gaussian_smoothing(self, sigma = 1):
        '''
        Perform image gaussian smoothing. Not sure if it is necessary with our images (are they noisy?)

        Modifies: 000_channels
        '''
        for channel in self.channel_names:
            self.layers['000_channels'][channel] = filters.gaussian(self.layers['000_channels'][channel],sigma = sigma)

    def segmentation_channel_intensity_based(self, channel, threshold_method = 'otsu', spot_sigma = 0, outline_sigma = 0, use_roi_mask = False, fit_threshold_to_signal_mask = 0, inherit_threshold = -1):
        '''
        Do intensity-based image segmentation. To avoid SIM artifacts, it computes a preliminary threshold in the Z max projection.
        From there, two results:
            - The original image is thresholded to remove lower intensity signal, and the thresholded image is segmented (Otsu-Voronoi)
            - The original image is segmented (Otsu-Voronoi), and then segmented areas are trimmed based on a mask of the thresholded image 
        For now, Otsu thresholding is the only option, but could be expanded to custom thresholds.
        Segmentation algorithm could also be changed, but Otsu-Voronoi is widely used.

        Identifier: 003

        Arguments:
        - channel: string. Name of channel to be segmented (channel names given when making the SIM_segmentation object).
        - threshold_method: string. Default 'otsu'. Threshold method to be applied. For now, only 'otsu', but algorithm could be extended to others.
        - spot_sigma: float. Standard deviation of the 3D gaussian filter applied to find local maxima.
        - outline_sigma: float. Standard deviation of the 3D gaussian filter applied to Voronoi + watershed segmentation from local maxima.
        - use_roi_mask: boolean. Default False. If True, will exclude any signal outside '001_roi_mask', a binary mask obtained during truncation.
                Only applicable if truncation is not manual.
        - fit_threshold_to_signal_mask: float. Default 0. If above 0, ratio of background pixels that are positive pixel outside binary mask
                obtained during truncation. Only applicable if truncation is not manual.
                Recommended value above 2 (which means minimum 2 background pixels must be negative for every background pixel that is considered positive).
        - inherit_threshold: float. Default 0. If above 0, intensity threshold value to be applied. Overrides intensity-based thresholding methods.
        '''
        def find_labels_maxima(img, segmented_img, spot_sigma):
            '''
            Compute local intensity maxima that were used to identify segmented regions with Otsu-Voronoi thresholding.
            Returns a pandas data frame with region label, position of maxima used as seed and region centroid.
            '''
            blurred_img = cle.gaussian_blur(img, sigma_x=spot_sigma, sigma_y=spot_sigma, sigma_z=spot_sigma)
            local_maxima = cle.detect_maxima_box(blurred_img, radius_x=0, radius_y=0, radius_z=0) # Keeps indicating size for some reason???
            local_maxima = np.asarray(local_maxima)
            local_maxima[local_maxima > 0] = 1
            local_maxima = (local_maxima*segmented_img).astype(int)
            local_maxima = pd.DataFrame(regionprops_table(local_maxima, properties = ('label','centroid')))
            local_maxima.rename(columns = {'centroid-0':'maxima_z','centroid-1':'maxima_y','centroid-2':'maxima_x'}, inplace = True)
            centroids = pd.DataFrame(regionprops_table(segmented_img, properties = ('label','centroid')))
            centroids.rename(columns = {'centroid-0':'centroid_z','centroid-1':'centroid_y','centroid-2':'centroid_x'}, inplace = True)
            maxima_and_centroids = pd.merge(local_maxima,centroids,on='label')
            ## We may lose some labels based on lack of local maxima reported in this manner. It should be equivalent, but is not the same package so could have slight changes.
            ## From experience, the changes are missing 1 label, so simply remove that label with update_labels_maxima later.
            for label in range(1,np.max(segmented_img)+1):
                if label not in maxima_and_centroids.loc[:,'label']:
                    segmented_img[segmented_img == label] = 0
            return segmented_img, maxima_and_centroids
        
        def voronoi_manual_labeling(img, spot_sigma = 0, outline_sigma = 0, threshold = 0):
            '''
            Slight modification from PyClEsperanto voronoi_otsu_labeling (technically will be the same as that function if threshold left untouched).
            '''
            blurred = cle.gaussian_blur(img, sigma_x=spot_sigma, sigma_y=spot_sigma, sigma_z=spot_sigma)
            detected_spots = cle.detect_maxima_box(blurred, radius_x=0, radius_y=0, radius_z=0)
            blurred = cle.gaussian_blur(img, sigma_x=outline_sigma, sigma_y=outline_sigma, sigma_z=outline_sigma)
            # if threshold_method == 'otsu':
            #     binary = cle.threshold_otsu(blurred)
            # elif threshold_method == 'moments':
            #     threshold = moments_thresholding(np.asarray(cle.maximum_z_projection(blurred))*255)/255
            #     binary = copy.copy(img)
            #     binary[binary < threshold] = 0
            #     binary[binary >= threshold] = 1
            # elif type(threshold_method) == float or type(threshold_method) == int:
            #     threshold = threshold_method
            #     binary = copy.copy(img)
            #     binary[binary < threshold] = 0
            #     binary[binary >= threshold] = 1
            try:
                threshold = float(threshold)
            except:
                raise Exception('threshold_method choice has been deprecated. voronoi_manual_labeling expects a number now.')
            binary = copy.copy(img)
            binary[binary < threshold] = 0
            binary[binary >= threshold] = 1                
            selected_spots = cle.binary_and(binary, detected_spots)
            voronoi_diagram = cle.masked_voronoi_labeling(selected_spots, binary)
            return voronoi_diagram

        try:
            self.channels_threshold_seg
        except:
            self.channels_threshold_seg = {}
        dict_keys_check = ['003_segmented_raw_channels','003_segmented_threshold_channels','003_segmented_trim_channels','003_roi_masks']
        for key in dict_keys_check:
            try:
                self.layers[key]
            except:
                self.layers[key] = {}
        dict_keys_check = ['003_segmented_raw_maxima','003_segmented_threshold_maxima','003_segmented_trim_maxima']
        for key in dict_keys_check:
            try:
                self.tables[key]
            except:
                self.tables[key] = {}
        # Segmentation of the raw image, no image thresholding applied (except roi mask, and optional)
        img = copy.copy(self.layers['000_channels'][channel])
        max_z_proj = np.array(cle.maximum_z_projection(img))
        # Best results with SIM artifacts has relied on obtaining an initial mask to focus only on the brightest signals (another option would be to exclude segmented signals above or below a brighter and bigger one...).
        ## Get initial threshold cut-off in max-Z projection that will later be applied to the whole stack.
        if inherit_threshold > 0:
            threshold = inherit_threshold
        elif threshold_method == 'otsu':
            threshold = filters.threshold_otsu(max_z_proj)
        elif threshold_method == 'moments':
            threshold = moments_thresholding(max_z_proj*255)/255
        ## Case that want to remove signal outside of the cell, but also reduce in ROI
        if fit_threshold_to_signal_mask > 0:
            ## fit mask fine-tuning: 0 is excluded in both thresholded and roi_mask, 1 is roi_mask only, 2 is thresholded only, 3 is both included.
            ## Should make threshold more strict (closer to 1) until thresholded-only is minimal compared to the roi_mask.
            ## Minimal is tricky to establish. For now, a background balance (only applied to pixels outside roi_mask, for every pixel above threshold there are N below threshold).
            thresholded_maxzproj = copy.copy(max_z_proj)
            thresholded_maxzproj[max_z_proj < threshold] = 0
            thresholded_maxzproj[max_z_proj >= threshold] = 2
            fit_match = thresholded_maxzproj + self.layers['001_signal_mask']
            print('INITIAL THRESHOLD: ' + str(threshold))
            print('BEFORE ADJUSTING (background - no signal; mask - no signal; background - signal; mask - signal)')
            print(fit_match.shape, np.sum(fit_match == 0), np.sum(fit_match == 1), np.sum(fit_match == 2), np.sum(fit_match == 3))
            print('BACKGROUND, NO SIGNAL TO SIGNAL RATIO: ' + str((1+np.sum(fit_match == 0))/(1+np.sum(fit_match == 2))))
            while ((1+np.sum(fit_match == 0))/(1+np.sum(fit_match == 2))) < fit_threshold_to_signal_mask:  #Add pseudo-counts, although can handle DivideByZero warning.
                threshold += 0.0001
                thresholded_maxzproj[max_z_proj < threshold] = 0
                fit_match = thresholded_maxzproj + self.layers['001_signal_mask']
                if threshold > 1:
                    threshold == 1
                    break
            print('AFTER ADJUSTING (background - no signal; mask - no signal; background - signal; mask - signal)')
            print(fit_match.shape, np.sum(fit_match == 0), np.sum(fit_match == 1), np.sum(fit_match == 2), np.sum(fit_match == 3))
            print('BACKGROUND, NO SIGNAL TO SIGNAL RATIO: ' + str((1+np.sum(fit_match == 0))/(1+np.sum(fit_match == 2))))
            print('FINAL THRESHOLD: ' + str(threshold))
        self.channels_threshold_seg[channel] = threshold

        if threshold_method == 'min':
            threshold = 0
            thresholded_maxzproj = copy.copy(max_z_proj)
            thresholded_maxzproj[max_z_proj < threshold] = 0
            thresholded_maxzproj[max_z_proj >= threshold] = 2
            fit_match = thresholded_maxzproj + self.layers['001_signal_mask']
            print('INITIAL THRESHOLD: ' + str(threshold))
            print('BEFORE ADJUSTING (background - no signal; mask - no signal; background - signal; mask - signal)')
            print(fit_match.shape, np.sum(fit_match == 0), np.sum(fit_match == 1), np.sum(fit_match == 2), np.sum(fit_match == 3))
            print('BACKGROUND, NO SIGNAL TO SIGNAL RATIO: ' + str((1+np.sum(fit_match == 0))/(1+np.sum(fit_match == 2))))
            while ((1+np.sum(fit_match == 0))/(1+np.sum(fit_match == 2))) < fit_threshold_to_signal_mask:  #Add pseudo-counts, although can handle DivideByZero warning.
                threshold += 0.000015
                thresholded_maxzproj[max_z_proj < threshold] = 0
                fit_match = thresholded_maxzproj + self.layers['001_signal_mask']
                if threshold > 1:
                    threshold == 1
                    break
            print('AFTER ADJUSTING (background - no signal; mask - no signal; background - signal; mask - signal)')
            print(fit_match.shape, np.sum(fit_match == 0), np.sum(fit_match == 1), np.sum(fit_match == 2), np.sum(fit_match == 3))
            print('BACKGROUND, NO SIGNAL TO SIGNAL RATIO: ' + str((1+np.sum(fit_match == 0))/(1+np.sum(fit_match == 2))))
            print('FINAL THRESHOLD: ' + str(threshold))
        ## Option 0: default segmentation on the given image stack. Threshold cut-off is used as a regular intensity-based thresholding.
        self.layers['003_segmented_raw_channels'][channel] = voronoi_manual_labeling(img, spot_sigma, outline_sigma, threshold = threshold)
        self.layers['003_segmented_raw_channels'][channel] = label(self.layers['003_segmented_raw_channels'][channel])
        self.layers['003_segmented_raw_channels'][channel],self.tables['003_segmented_raw_maxima'][channel] = find_labels_maxima(img, self.layers['003_segmented_raw_channels'][channel], spot_sigma)
        if use_roi_mask:
            self.layers['003_segmented_raw_channels'][channel] = label(self.layers['003_segmented_raw_channels'][channel]*self.layers['006_roi_mask_pseudo3D']).astype(int)
            # Get rid of any labels that do not have a local maxima in region of interest mask
            update_labels_maxima(self,channel,'raw')
            for trim_label in range(1,np.max(self.layers['003_segmented_raw_channels'][channel])+1):
                if trim_label not in self.tables['003_segmented_raw_maxima'][channel]['label'].values:
                    self.layers['003_segmented_raw_channels'][channel][self.layers['003_segmented_raw_channels'][channel] == trim_label] = 0
        ## Option 1: double thresholding segmentation. Use threshold cut-off to turn every pixel below threshold to threshold value.
        ##      Use raised-minimum stack for segmentation.
        ##      FROM EXPERIENCE, does not work well as pure image segmentation, too strict. BUT, can help its implementation to filter out local maxima.
        img_thresholded = copy.copy(self.layers['000_channels'][channel])
        img_thresholded[img_thresholded < threshold] = threshold
        self.layers['003_segmented_threshold_channels'][channel] = voronoi_manual_labeling(img_thresholded, spot_sigma, outline_sigma, threshold = threshold)
        self.layers['003_segmented_threshold_channels'][channel] = label(self.layers['003_segmented_threshold_channels'][channel])
        self.layers['003_segmented_threshold_channels'][channel],self.tables['003_segmented_threshold_maxima'][channel] = find_labels_maxima(img_thresholded, self.layers['003_segmented_threshold_channels'][channel], spot_sigma)
        if use_roi_mask:
            self.layers['003_segmented_threshold_channels'][channel] = label(self.layers['003_segmented_threshold_channels'][channel]*self.layers['006_roi_mask_pseudo3D']).astype(int)
            # Get rid of any labels that do not have a local maxima in region of interest mask
            update_labels_maxima(self,channel,'threshold')
            for trim_label in range(1,np.max(self.layers['003_segmented_threshold_channels'][channel])+1):
                if trim_label not in self.tables['003_segmented_threshold_maxima'][channel]['label'].values:
                    self.layers['003_segmented_threshold_channels'][channel][self.layers['003_segmented_threshold_channels'][channel] == trim_label] = 0
        ## Option 2: trimming base segmentation with initial threshold. Use full-reach Voronoi tesselation, and remove regions outside cut-off mask from raw image.
        ##      Intersect binary mask with raw stack LABELED segmentation (only applying sigmas related to the intensity-based segmentation).
        ##      If a region gets fragmented in several non-linked regions, exclude regions that are not connected to a segmentation local maxima.
        trim_mask = copy.copy(self.layers['000_channels'][channel])
        trim_mask[trim_mask < threshold] = 0
        trim_mask[trim_mask >= threshold] = 1
        if use_roi_mask:
            trim_mask = (trim_mask*self.layers['006_roi_mask_pseudo3D']).astype(int)
        self.layers['003_roi_masks'][channel] = trim_mask
        img_segmented_trimmed = voronoi_manual_labeling(img, spot_sigma, outline_sigma, threshold = 0)
        img_segmented_trimmed = label(img_segmented_trimmed)
        img_segmented_trimmed,img_segmented_maxima = find_labels_maxima(self.layers['000_channels'][channel], img_segmented_trimmed, spot_sigma)
        img_segmented_trimmed = label((img_segmented_trimmed*trim_mask).astype(int))    ## Has to relabel to give unique ID to split regions after trimming.
        self.layers['003_segmented_trim_channels'][channel] = img_segmented_trimmed
        self.tables['003_segmented_trim_maxima'][channel] = img_segmented_maxima
        update_labels_maxima(self,channel,'trim')
        # Get rid of trimmed split labels that do not have a local maxima in region of interest mask
        for trim_label in range(1,np.max(self.layers['003_segmented_trim_channels'][channel])+1):
            if trim_label not in self.tables['003_segmented_trim_maxima'][channel]['label'].values:
                self.layers['003_segmented_trim_channels'][channel][self.layers['003_segmented_trim_channels'][channel] == trim_label] = 0
        self.time_log['Segmentation_intensity'] = {}
        self.time_log['Segmentation_intensity'][channel] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, intensity-based segmentation of {}: {}'.format(channel, self.time_log['Segmentation_intensity'][channel]))
        self.start_time = time.time()

    def remove_small_regions(self, channel, keyword, min_size_filter = 0):
        '''
        Filter out segmented regions smaller than a certain size.

        Modifies:
        - 003

        Arguments:
        - channel: string. Name of channel to remove small regions from segmentation.
        - keyword: string. Unique string to retrieve specific segmentation results for that channel.
                Usable now, 'threshold' and 'trim'.
                Example: 'threshold' will identify '003_segmented_threshold_channels' and '003_segmented_threshold_maxima'.
        '''
        ## Find substring in a numpy array, use numpy find function (returns index of substring if found, -1 if substring not found in that element).
        ## Since we only want to find key with substring, turn into boolean (-1 turned to False, >-1 turned to True).
        ## flatnonzero returns indices of array that are not zero (False is zero).
        segmented_key = find_key_from_pattern(self, keyword,'layers')
        self.layers[segmented_key][channel] = morphology.remove_small_objects(self.layers[segmented_key][channel], min_size_filter)
        update_labels_maxima(self, channel, keyword)
        self.time_log['remove_small_regions'] = {}
        self.time_log['remove_small_regions'][channel] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, remove small regions of {}: {}'.format(channel, self.time_log['remove_small_regions'][channel]))
        self.start_time = time.time()
    
    def select_segmentation(self, keyword):
        '''
        Specify a segmentation from SIM_segmentation object to be used in following analysis.

        Identifier: 004

        Arguments:
        - keyword: string. Unique string to retrieve specific segmentation results for analysis.
        '''
        segmented_key = find_key_from_pattern(self, keyword,'layers')
        maxima_key = find_key_from_pattern(self, keyword,'tables')
        self.layers['004_seg_mask'] = {}
        self.layers['004_segmented_labeled'] = {}
        self.tables['004_segmented_regions'] = {}
        self.tables['004_segmented_maxima'] = {}
        for channel in self.channel_names:
            self.layers['004_seg_mask'][channel] = copy.copy(self.layers[segmented_key][channel])
            self.layers['004_seg_mask'][channel][self.layers['004_seg_mask'][channel] > 0] = 1
            self.layers['004_segmented_labeled'][channel] = self.layers[segmented_key][channel] ## Do not label again, let the original labels be there as a reference of what was removed in case it is run with different filters.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ## Used to retrieve axis_minor_length and axis_major_length, but sometimes lead to sqrt of negative and stopped working.
                self.tables['004_segmented_regions'][channel] = pd.DataFrame(regionprops_table(self.layers['004_segmented_labeled'][channel],
                        intensity_image=self.layers['000_channels'][channel],
                        properties = ['label','area','area_bbox','area_filled','bbox','centroid','centroid_weighted',
                            'equivalent_diameter_area','euler_number','extent','inertia_tensor','inertia_tensor_eigvals','intensity_max',
                            'intensity_mean','intensity_min','moments','moments_central','moments_normalized','moments_weighted',
                            'moments_weighted_central','moments_weighted_normalized','slice','solidity']))
            self.tables['004_segmented_maxima'][channel] = self.tables[maxima_key][channel]
            update_labels_maxima(self, channel, '004_segmented')
        self.time_log['label_seg'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, standardizing labels of chosen segmentation: {}'.format(self.time_log['label_seg']))
        self.start_time = time.time()
    
    def filter_common_maxima(self, initialseg_keyword, newmaxima_keyword):
        '''
        Takes one set of segmentation results (regions plus maxima table) and another table of local maxima.
        Returns the regions of the segmentation that present a common maxima with the new table.
        
        Identifier: 005

        Arguments:
        - initialseg_keyword: string. Unique string to retrieve specific segmentation results for analysis.
        - newmaxima_keyword: string. Unique string to retrieve specific local maxima table independent of initialseg_keyword.
        '''
        segmented_key = find_key_from_pattern(self, initialseg_keyword,'layers')
        maxima_key = find_key_from_pattern(self, initialseg_keyword,'tables')
        newmaxima_key = find_key_from_pattern(self, newmaxima_keyword,'tables')
        if maxima_key == newmaxima_key:
            raise Exception('maxima_key and newmaxima_key identical, segmentation would not be filtered.')
        name_layer_dict = '005_{}_filtered_by_{}'.format(initialseg_keyword,newmaxima_keyword)
        self.layers[name_layer_dict] = {}
        self.tables[name_layer_dict+'_maxima'] = {}
        for channel in self.channel_names:
            ## First, copy segmentation and new maxima with similar entry names, and update new maxima labels to fit segmentation.
            self.layers[name_layer_dict][channel] = copy.copy(self.layers[segmented_key][channel])
            self.tables[name_layer_dict+'_maxima'][channel] = copy.copy(self.tables[newmaxima_key][channel])
            update_labels_maxima(self, channel, name_layer_dict)
            ## Remove regions that do not have local maxima now. It is possible that local maxima is not in identical to original, or multiple local maxima to single region...
            for label in range(1,np.max(self.layers[name_layer_dict][channel])+1):
                if label not in self.tables[name_layer_dict+'_maxima'][channel].loc[:,'label'].values:
                    self.layers[name_layer_dict][channel][self.layers[name_layer_dict][channel] == label] = 0
            ## Change local maxima of the filtered segmentation to the original, and update to remove lost local maxima after filtering.
            self.tables[name_layer_dict+'_maxima'][channel] = copy.copy(self.tables[maxima_key][channel])
            update_labels_maxima(self, channel, name_layer_dict)
        self.time_log['filter_two_seg'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, filtering one segmentation based on local maxima of another: {}'.format(self.time_log['filter_two_seg']))
        self.start_time = time.time()
    
    def make_spatial_mask(self, prism_mode = False):
        '''
        Make pseudo-3D mask, for spatial purposes. Full 3D is too challenging without proper reference (DAPI or nuclear envelope). But can get an approximate 'shadow' per dim.

        - cylinder_mode: boolean (default, False). If True, will use same shape as self.layers['001_roi_mask'] and expand to the full stack. Will not shrink based on signal

        Identifier: 006
        '''
        # First, remove high signal outside ROI (for instance, signal coming from another cell that is in the edge of the crop).
        # Second, make max projections in each dimension of all channels without high signal outside ROI. Turn projections into mask based on original thresholding of max_z_projection
        # Third, make pseudo-3D ROI by multiplying all these max projection masks, extended in their corresponding missing dimension.
    #    roi_mask_isolated_projs = {}
        roi_mask_prism = np.repeat(self.layers['001_roi_mask'][np.newaxis,:,:], self.layers['000_channels'][self.channel_names[0]].shape[0], axis=0)
        if prism_mode == True:
            roi_mask_pseudo3D = roi_mask_prism
        else:
            isolated_mask = np.ones(self.layers['000_channels'][self.channel_names[0]].shape)
            for channel in self.trunc_channels:
                isolated_signal = copy.copy(self.layers['000_channels'][channel])
                ## Original image is 16-bit, but it is possible after going through some functions (example, anisotropy_correction) that it turns into 0-to-1. Have to correct thresholding from truncation.
                if np.max(isolated_signal) <= 1:
                    threshold_for_mask = self.channels_threshold_trunc[channel]/65535
                else:
                    threshold_for_mask = self.channels_threshold_trunc[channel]
                ## Ignore signal outside ROI,  adjusted to being a Z-stack unlike before.
                isolated_signal[np.logical_and(np.invert(np.array(roi_mask_prism, dtype = bool)), self.layers['000_channels'][channel] > threshold_for_mask)] = threshold_for_mask
                isolated_mask_temp = np.zeros(isolated_signal.shape)
                isolated_mask_temp[isolated_signal <= threshold_for_mask] = 0
                isolated_mask_temp[isolated_signal > threshold_for_mask] = 1
                isolated_mask = isolated_mask*isolated_mask_temp
            roi_mask_pseudo3D = np.asarray(morphology.convex_hull_image(isolated_mask))*roi_mask_prism
        roi_mask_pseudo3D_core = copy.copy(roi_mask_pseudo3D)
        roi_mask_pseudo3D_rim = copy.copy(roi_mask_pseudo3D)
        footprint = np.ones((3,3))  ## This should give a very narrow contour of the cell
        roi_mask_pseudo3D_core = morphology.skeletonize(roi_mask_pseudo3D_core)
        for z_slice in range(roi_mask_pseudo3D.shape[0]):
            roi_mask_pseudo3D_rim[z_slice,:,:] = morphology.binary_erosion(roi_mask_pseudo3D_rim[z_slice,:,:],footprint)
        ## For a broader identification, to say "this is the rim". Should not be used together with dist_to_core_and_rim. More to put a label of "rim" or "core".
        #for z_slice in range(roi_mask_pseudo3D.shape[0]):
        #    roi_mask_pseudo3D_core[z_slice,:,:] = self.binary_morphology_fast(np.squeeze(roi_mask_pseudo3D_core[z_slice,:,:]), mode = 'erosion', morph_window_length = 100)
        roi_mask_pseudo3D_rim = np.logical_xor(roi_mask_pseudo3D.astype(bool),roi_mask_pseudo3D_rim.astype(bool)).astype(int)
        self.layers['006_roi_mask_pseudo3D'] = roi_mask_pseudo3D
        self.layers['006_roi_mask_pseudo3D_core'] = roi_mask_pseudo3D_core
        self.layers['006_roi_mask_pseudo3D_rim'] = roi_mask_pseudo3D_rim
        self.time_log['make_spatial_mask'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, making spatial mask: ' + self.time_log['make_spatial_mask'])
        self.start_time = time.time()

    def dist_to_core_and_rim(self):
        '''
        For each signal centroid (no need to also do maxima, probably negligible), calculate distance to origin (centroid of spatial mask) and rim (closest point of the rim).
        Only applies to selected segmentation (before there are too many, could be confusing).

        Identifier: 007
        '''
        ## Need every column to have same column names in order to be fed to pairwise distance.
        core_coords = np.where(self.layers['006_roi_mask_pseudo3D_core'] > 0)
        core_coords = pd.DataFrame(zip(*core_coords), columns = np.array(['coord_z','coord_y','coord_x']))
        #center_coord = pd.DataFrame(regionprops_table(self.layers['006_roi_mask_pseudo3D_core'], properties=['centroid']))
        #if center_coord.shape[0] > 1:
        #    raise Exception('spatial mask has more than one region.')
        rim_coords = np.where(self.layers['006_roi_mask_pseudo3D_rim'] > 0)
        rim_coords = pd.DataFrame(zip(*rim_coords), columns = np.array(['coord_z','coord_y','coord_x']))
        #self.tables['007_dist_core_to_rim'] = pd.DataFrame(pairwise_distances(rim_coords,core_coords), index = list(range(rim_coords.shape[0])), columns = list(range(core_coords.shape[0])))
        self.tables['007_dist_to_core_and_rim'] = {}
        for channel in self.channel_names:
            reg_coords = self.tables['004_segmented_maxima'][channel].loc[:,['centroid_z','centroid_y','centroid_x']]
            self.tables['007_dist_to_core_and_rim'][channel] = pd.concat([pd.DataFrame(pairwise_distances(core_coords,reg_coords).min(axis=0), index = self.tables['004_segmented_maxima'][channel].loc[:,'label'], columns = ['core_dist']),
                                                                          pd.DataFrame(pairwise_distances(rim_coords,reg_coords).min(axis=0), index = self.tables['004_segmented_maxima'][channel].loc[:,'label'], columns = ['rim_dist'])],
                                                                          axis=1)
        self.time_log['dist_to_core_and_rim'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, calculating distance of selected segmentation to spatial center and rim: {}'.format(self.time_log['dist_to_core_and_rim']))
        self.start_time = time.time()

    def save_segmentation_layers(self, prefix = '', suffix = '', only_selected_seg = True):
        '''
        Save all images and tables stored in the object so far.

        only_selected_seg: boolean (default, True). If True, will only save the final segmentation selected, and not all computed during intensity_based_segmentation step.
        '''
        ### Should it have a saved description (xml).
        save_dir = os.path.join(self.RES_IMAGE_PATH,'layers')
        check_to_make_dir(save_dir)
        for layer_key in self.layers.keys():
            if only_selected_seg:
                if any(['_threshold_' in layer_key, '_raw_' in layer_key, '_trim_' in layer_key]):
                    continue
            if type(self.layers[layer_key]) == dict:
                for channel in self.channel_names:
                    try:
                        f = gzip.GzipFile('{}/{}{}{}_{}.npy.gz'.format(save_dir, prefix, layer_key, suffix, channel), "w")
                        np.save(file=f, arr=self.layers[layer_key][channel])
                        f.close()
                    except:
                        pass
            else:
                f = gzip.GzipFile('{}/{}{}{}.npy.gz'.format(save_dir, prefix, layer_key, suffix), "w")
                np.save(file=f, arr=self.layers[layer_key])
                f.close()
        save_dir = os.path.join(self.RES_IMAGE_PATH,'tables')
        check_to_make_dir(save_dir)
        for table_key in self.tables.keys():
            if only_selected_seg:
                if any(['_threshold_' in table_key, '_raw_' in table_key, '_trim_' in table_key]):
                    continue
            if type(self.tables[table_key]) == dict:
                for channel in self.channel_names:
                    try:
                        CSV_path = '{}/{}{}{}_{}.csv.gz'.format(save_dir,prefix,table_key,suffix,channel)
                        if not os.path.exists(CSV_path):
                            self.tables[table_key][channel].to_csv(CSV_path, sep = '\t', na_rep = 'NaN',index=True, index_label = 'INDEX',compression = 'gzip')
                    except:
                        pass
            else:
                CSV_path = '{}/{}{}{}.csv.gz'.format(save_dir,prefix,table_key,suffix)
                if not os.path.exists(CSV_path):
                    self.tables[table_key].to_csv(CSV_path, sep = '\t', na_rep = 'NaN',index=True, index_label = 'INDEX', compression = 'gzip')
        self.time_log['save_seg'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, saving segmentation layers and tables: {}'.format(self.time_log['save_seg']))
        self.start_time = time.time()