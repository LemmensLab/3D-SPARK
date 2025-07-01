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
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sklearn import get_config
import pyclesperanto_prototype as cle
import scipy
import seaborn as sns
import xml.etree.ElementTree as ET
import warnings
from tqdm import tqdm
import gzip
from QuantSIM.common import check_to_make_dir


class SIM_analysis:
    '''
    Python class to identify overlapping regions in two channels of SIM_segmentation object. Based on Robert Welch original script.
    
    Identifier: 101

    Arguments:
        SIM_segmentation: a SIM_segmentation object that contains segmentations to be worked on.
    '''
    def __init__(self, SIM_segmentation, channel_1, channel_2):
        self.seg = SIM_segmentation
        self.analysis_channels = [channel_1,channel_2]
        for channel in self.analysis_channels:
            if channel not in self.seg.channel_names:
                raise Exception('channels indicated for segmentation analysis are not present in SIM_segmentation object.')
        self.layers = {}
        self.tables = {}
        self.region_props = {}
        self.time_log = {}
        self.time_log['time-0'] = time.time()
        self.start_time = time.time()


    def int_list_centroids(self, properties):
        '''
        Function called by other class functions.
        Returns list of centroids (z,y,x) for a given set of region properties object.
        '''
        # Takes properties and returns tuple integer list of centroids
        intersection_centroids = []
        for intersecting_region in properties:
            intersection_centroids.append(intersecting_region.centroid)
        #Convert centroids to integer values for indexing
        intersection_centroids_int = []
        for centroid in intersection_centroids:
            z = int(centroid[0])
            y = int(centroid[1])
            x = int(centroid[2])
            intersection_centroids_int.append([z,y,x])
        return intersection_centroids_int
    

    def overlap_detection(self, min_size_overlap = 0):
        '''
        Find overlapping regions between two segmented channels.
        If channel_1 == channel_2, should find no overlaps even if regions are touching (except region with itself).

        Identifier: 102

        Parameters:
        - (taken from self):    self.seg.layers['004_segmented_labeled']
                                self.seg.tables['004_segmented_maxima']
        - make_CSV: boolean (default as False). If True, will generate CSV file with every row identified with a channel_1 label, and every column with a channel_2 label.
        
        Always saves table results in the SIM_analysis object.
        '''
        ### Will use a combination of overlapping labels to name the intersection region.
        ### Since it has to be integers, will concatenate label of channel 1 + label of channel 2.
        ### Have to find biggest label to define number of digits per label. Example:
        ###     label 1 in channel 1 + label 2 in channel 2 could be 12, but if another is label 1 + label 234, it is 1234 and where is the cutoff to identify original labels?
        ###     If labels of maximum 3 digits exist, the previous overlaps should have labels 1002 and 1234, and we know to cut at slice -3.
        union_mask = self.seg.layers['004_segmented_labeled'][self.analysis_channels[0]].astype('bool')*self.seg.layers['004_segmented_labeled'][self.analysis_channels[1]].astype('bool')
        max_label = max([max(self.seg.tables['004_segmented_maxima'][self.analysis_channels[0]].loc[:,'label']), max(self.seg.tables['004_segmented_maxima'][self.analysis_channels[1]].loc[:,'label'])])
        max_label = len(str(max_label))
        def intersection_label_slicing(label,split_index):
            return int(str(label)[:-split_index]),int(str(label)[-split_index:])
        vfunc_intersection_label_slicing = np.vectorize(intersection_label_slicing)
        intersection_labeled = union_mask*(self.seg.layers['004_segmented_labeled'][self.analysis_channels[0]]*(10**max_label)+self.seg.layers['004_segmented_labeled'][self.analysis_channels[1]]).astype(int)
        intersection_labeled = morphology.remove_small_objects(intersection_labeled, min_size_overlap)
        # Set up boolean table to mark overlaps between regions (all set to 0, or False).
        intersection_table = np.zeros((self.seg.tables['004_segmented_maxima'][self.analysis_channels[0]].shape[0],self.seg.tables['004_segmented_maxima'][self.analysis_channels[1]].shape[0]))
        intersection_table = pd.DataFrame(intersection_table, index = self.seg.tables['004_segmented_maxima'][self.analysis_channels[0]].loc[:,'label'], columns = self.seg.tables['004_segmented_maxima'][self.analysis_channels[1]].loc[:,'label'])
        ## Some properties cannot be acquired in intersection because some overlaps are so small it may be reported as 1D array or are less than 4 pixels (case of axis_major and axis_minor), but most are good.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intersection_regions = pd.DataFrame(regionprops_table(intersection_labeled, properties = ['label','area','area_bbox','area_filled','bbox','centroid','centroid_local',
                                                                                    'equivalent_diameter_area','euler_number','extent','inertia_tensor','inertia_tensor_eigvals',
                                                                                    'moments','moments_central','moments_normalized','slice','solidity']))
        if np.sum(intersection_labeled > 0) == 0:
            self.tables['102_overlap_bool'] = intersection_table.astype(bool)
            self.tables['102_intersection_regions'] = intersection_regions
            return None
        intersection_regionnames = pd.DataFrame(zip(*vfunc_intersection_label_slicing(intersection_regions['label'],max_label)), columns = self.analysis_channels)
        intersection_regions = pd.concat((intersection_regionnames, intersection_regions), axis = 1)
        labels_1 = intersection_regions[self.analysis_channels[0]]
        labels_2 = intersection_regions[self.analysis_channels[1]]
        ## Same channel problem: both columns have identical name! Using the column does not return a single pandas Series but a two-column-wide DataFrame.
        ## To solve that, assign to labels_1 and labels_2 the same pandas Series (one column)
        if self.analysis_channels[0] == self.analysis_channels[1]:
            labels_1 = labels_2 = intersection_regions[self.analysis_channels[0]].iloc[:,0]
        for label_1,label_2 in zip(labels_1,labels_2):
            intersection_table.loc[label_1,label_2] = 1
        self.max_label = max_label
        self.layers['102_union_mask'] = union_mask
        self.layers['102_segmented_labeled_all_intersections'] = intersection_labeled
        self.tables['102_overlap_bool'] = intersection_table.astype(bool)
        self.tables['102_intersection_regions'] = intersection_regions
        self.time_log['Overlap_detection'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, overlap detection ({} and {}): {}'.format(self.analysis_channels[0],self.analysis_channels[1], self.time_log['Overlap_detection']))
        self.start_time = time.time()


    def overlap_declumping(self, threshold = 0.25):
        '''
        Declumps overlapping regions by removing regions with an overlap area
        smaller than 25% of the maximum overlap in intersecting regions.
        Updates overlap_bool and repeats the process until no changes occur.
        '''
        ## For sure this can be done more compact, by transposing the 2D-matrix...
        # Start of overlap_declumping function
        original_overlap_bool = self.tables['102_overlap_bool'].copy()
        intersection_regions = self.tables['102_intersection_regions'].copy()

        def filter_overlaps(overlap_bool, intersection_regions):
            new_overlap_bool = overlap_bool.copy()
            # First check by rows
            for label_1 in overlap_bool.index.tolist():
                labels_2 = overlap_bool.columns[np.where(overlap_bool.loc[label_1,:])].tolist()
                if len(labels_2) > 1:
                    intersecting_regions = intersection_regions.loc[np.logical_and((intersection_regions[self.analysis_channels[0]] == label_1).to_numpy(),
                                                                                   np.array([temp in labels_2 for temp in intersection_regions[self.analysis_channels[1]]])),
                                                                :]
                    overlap_vols = intersecting_regions['area'].to_numpy()
                    max_vol = np.max(overlap_vols)
                    threshold_vol = max_vol * threshold
                    for label_2 in labels_2:
                        this_vol = intersecting_regions.loc[intersection_regions[self.analysis_channels[1]] == label_2,'area'].to_numpy()
                        if this_vol < threshold_vol:
                            new_overlap_bool.loc[label_1, label_2] = False

            # Then check by columns
            for label_2 in overlap_bool.columns.tolist():
                labels_1 = overlap_bool.index[np.where(overlap_bool.loc[:,label_2])].tolist()
                if len(labels_1) > 1:
                    intersecting_regions = intersection_regions.loc[np.logical_and(np.array([temp in labels_1 for temp in intersection_regions[self.analysis_channels[0]]]),
                                                                               (intersection_regions[self.analysis_channels[1]] == label_2).to_numpy()),
                                                                :]
                    overlap_vols = intersecting_regions['area'].to_numpy()
                    max_vol = np.max(overlap_vols)
                    threshold_vol = max_vol * threshold
                    for label_1 in labels_1:
                        this_vol = intersecting_regions.loc[intersection_regions[self.analysis_channels[0]] == label_1,'area'].to_numpy()
                        if this_vol < threshold_vol:
                            new_overlap_bool.loc[label_1, label_2] = False
            return new_overlap_bool

        changed = True
        iteration_count = 0

        while changed:
            iteration_count += 1
            filtered_overlap_bool = filter_overlaps(original_overlap_bool, intersection_regions)
            if filtered_overlap_bool.equals(original_overlap_bool):
                changed = False
            else:
                original_overlap_bool = filtered_overlap_bool.copy()
            #print(f"Iteration {iteration_count}: Changes detected." if changed else f"Iteration {iteration_count}: No changes detected, stopping.")

        self.tables['102_overlap_bool'] = original_overlap_bool
        self.tables['102_intersection_regions'] = intersection_regions

        #print(f"Overlap declumping complete after {iteration_count} iterations.")
        self.time_log['declumping'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, overlap decclumping ({} vs {}), {} iteration(s): {}'.format(self.analysis_channels[0],self.analysis_channels[1],
                                                                                            str(iteration_count),self.time_log['declumping']))
        self.start_time = time.time()


    def neighbor_distance(self, channels_ref = ['centroid','centroid']):
        '''
        Calculates distance of regions between channels based on a point-based metric.
        If channel_1 == channel_2, then it calculates distances of regions in the same channel.
        Surface-to-surface not recommended due to the number of points to be used for calculations (very slow, may run out of memory).

        Identifier: 103

        Parameters:
        - (taken from self):    self.seg.layers['004_segmented_labeled']
                                self.seg.tables['004_segmented_maxima']
        - channels_ref: list [channel_1_ref, channel_2_ref]. What reference of segmented regions of each channel to use for calculating distances.
            Keywords: centroid, maxima, surface.
        
        Always saves table results in the SIM_analysis object.
        '''
        points_for_dists = {} # coordinates of points to calculate distances in each channel
        labels = {} # Contains labels name in each channel (one row per region per channel)
        edgelabels = {} # Only for "surface". Contains all edge pixel coordinates label (one row per edge pixel per region per channel).

        ## Check if channels are centroid, maxima or surface based on and calculate points accordingly.
        for i,channel_ref in enumerate(channels_ref):
            labels[self.analysis_channels[i]] = self.seg.tables['004_segmented_maxima'][self.analysis_channels[i]].loc[:,'label']
            if channel_ref == 'centroid' or channel_ref == 'maxima':
                channel_center_cols = [channel_ref+i for i in['_z','_y','_x']]
                points_for_dists[self.analysis_channels[i]] = np.asarray(self.seg.tables['004_segmented_maxima'][self.analysis_channels[i]].loc[:,channel_center_cols])
            elif channel_ref == 'surface':
                boundaries = find_boundaries(self.seg.layers['004_segmented_labeled'][self.analysis_channels[i]], mode = 'inner')*self.seg.layers['004_segmented_labeled'][self.analysis_channels[i]]
                edgelabels_index = np.where(boundaries > 0)
                points_for_dists[self.analysis_channels[i]] = np.asarray(list(zip(*edgelabels_index)))
                edgelabels[self.analysis_channels[i]] = np.asarray([boundaries[index] for index in zip(*edgelabels_index)])
        ## Calculate distances per pixel coordinate acquired in each channel.
        dists = pairwise_distances(points_for_dists[self.analysis_channels[0]], points_for_dists[self.analysis_channels[1]])
        dists_df = pd.DataFrame(dists)
        ## For "surface", compress distances so it is a single distance per label (closest point). For "centroid" or "maxima", just add labels as index.
        for i,channel_ref in enumerate(channels_ref):
            if i == 1:
                dists_df = dists_df.T # We need that evaluated channel is the Y-axis (Index, along rows). For channel_2 that is along columns, transpose DataFrame.
            if channel_ref == 'surface':
                dists_df.index = edgelabels[self.analysis_channels[i]] ## Multiple rows could have same label, for grouping. If it does not work, try MultiIndex, or pd.Index
                dists_df = dists_df.groupby(level = 0).min() ## This should check edgelabels index, group rows by equal label and get minimum value per equal index.
            else:
                dists_df.index = labels[self.analysis_channels[i]]
            if i == 1:
                dists_df = dists_df.T # When assessing channel_2, flip back DataFrame.
        self.tables['103_neighbor_{}_to_{}_dist'.format(channels_ref[0],channels_ref[1])] = dists_df
        self.time_log['neighbor_dist'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, distance between neighbors ({} vs {}): {}'.format(self.analysis_channels[0],self.analysis_channels[1],self.time_log['neighbor_dist']))
        self.start_time = time.time()

    def neighbor_closest_distance(self):
        '''
        LEGACY: good to check chunk-by-chunk calculations of closeness between segmentations.

        Calculates distance of regions between channels based on closest edge (more computational intensive than neighbor_center_distance).
        If channel_1 == channel_2, then it calculates distances of regions in the same chanel.

        Identifier: 104

        Parameters:
        - (taken from self):    self.seg.layers['004_segmented_labeled']
                                self.seg.tables['004_segmented_maxima']
        - make_CSV: boolean (default as False). If True, will generate CSV file with every row identified with a channel_1 label, and every column with a channel_2 label.
        
        Always saves table results in the SIM_analysis object.
        '''
        #### CHANGE SO IT GOES BY CHUNKS
        labels = {} # Contains labels name in each channel (one entry per channel)
        edgelabels = {} # Contains all edge pixel coordinates together with corresponding label (one entry per channel)
        colnames = ['label','coord_z','coord_y','coord_x']
        for channel in self.analysis_channels:
            labels[channel] = self.seg.tables['004_segmented_maxima'][channel].loc[:,'label']
            boundaries = find_boundaries(self.seg.layers['004_segmented_labeled'][channel], mode = 'inner')*self.seg.layers['004_segmented_labeled'][channel]
            edgelabels_index = np.where(boundaries > 0)
            edgelabels_label = np.asarray([boundaries[index] for index in zip(*edgelabels_index)])
            edgelabels[channel] = pd.DataFrame(zip(edgelabels_label,*edgelabels_index), columns = colnames)
        labels_index = {}   # Contains index of edges per label in edgelabels
        for channel in self.analysis_channels:
            labels_index[channel] = {}
            for label_in_channel in labels[channel]:
                labels_index[channel][label_in_channel] = np.array(edgelabels[channel].index)[np.where(edgelabels[channel]['label'] == label_in_channel)]
        closest_dists = pd.DataFrame(np.empty((len(labels[self.analysis_channels[0]]),len(labels[self.analysis_channels[1]]))), index = labels[self.analysis_channels[0]], columns = labels[self.analysis_channels[1]]) ## Empty, filled later
        ### pairwise distances kills the kernel, probably too many to calculate in one go. Instead, run algorithm in all possible pairs.
        #edge_dists = pairwise_distances(np.asarray(edgelabels[self.analysis_channels[0]].loc[:,colnames[1:4]]), np.asarray(edgelabels[self.analysis_channels[1]].loc[:,colnames[1:4]]))
        edgelabels_1 = edgelabels[self.analysis_channels[0]].iloc[:,[1,2,3]]
        edgelabels_2 = edgelabels[self.analysis_channels[1]].iloc[:,[1,2,3]]
        for label_1 in tqdm(labels[self.analysis_channels[0]]):
            edgelabels_1 = edgelabels[self.analysis_channels[0]].iloc[labels_index[self.analysis_channels[0]][label_1],[1,2,3]]
            edgelabels_2 = edgelabels[self.analysis_channels[1]].iloc[:,[1,2,3]]
            edge_dists = pd.DataFrame(pairwise_distances(np.asarray(edgelabels_1), np.asarray(edgelabels_2)))
            for label_2 in labels[self.analysis_channels[1]]:
                closest_dists.loc[label_1,label_2] = np.min(edge_dists.iloc[:,labels_index[self.analysis_channels[1]][label_2]].values)
        self.tables['104_neighbor_closest_dist'] = closest_dists
        self.time_log['neighbor_closest_dist'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, minimum distance between neighbors ({} vs {}): {}'.format(self.analysis_channels[0],self.analysis_channels[1],self.time_log['neighbor_closest_dist']))
        self.start_time = time.time()


    def two_channel_overlap_regions(self, overlap_stoich = (1,1)):
        '''
        Retrieve region labels that overlap with a defined stoichiometry between two channels. Restricted to stoichiometries (0,0), (1,1), (2,1) and (1,2).

        Identifier: 105

        Parameters:
        - (taken from self):    tables['102_overlap_bool']
        - overlap_stoich: tuple (default = (1,1)). Overlap stoichiometry to be retrieved. Examples:
                                (0,0): non-overlapping regions in each channel.
                                (1,1): overlap involving two regions. 1 region in channel_1 overlaps with 1 region in channel_2 & 1 region in channel_2 overlaps with 1 region in channel_1.
                                (2,1): overlap involving three regions. 1 region in channel_1 overlaps with 2 regions in channel_2 & 2 regions in channel_2 overlap with 1 region in channel_1.
                                (1,2): overlap involving three regions. 2 regions in channel_1 overlap with 1 region in channel_2 & 1 region in channel_2 overlaps with 2 regions in channel_1.
        '''
        if self.analysis_channels[0] == self.analysis_channels[1]:
                ## Special case: evaluating a channel against itself. Running through this function may be useful to get some summary plots later on.
                ## However, it may cause confusion because it overlaps 100% with itself and fits better as a (0,0) situation. So, always force to be that.
                overlap_stoich = (0,0)
        overlap_stoich = np.array(overlap_stoich)
        if all(overlap_stoich > 1) or any(overlap_stoich < 0) or sum(overlap_stoich == 0) == 1:
            raise Exception('Only stoichiometries (0,0), (1,1), (2,1) or (1,2) are valid.')
        
        ### Access data by row or column, looking for entries that fulfill n, and then check columns and rows to make sure that fulfill 1...
        ### Depending on what is found, return intersection region, regions in channel_1 and regions in channel_2
        ### applying functions along axis in pandas not like R! axis = 0 means applying function THROUGH rows (so get one result per column) and axis = 1 is THROUGH columns.
        ### In R, apply axis = 1 is result per row, and axis = 2 is result per column. You could say pandas and R use axis = 1 to apply function per row, but differ in numbering the other axis (easier to remember?)
        ### axis for pd.concat is technically the same as pd.function by axis. axis = 0 does row binding (same column number), axis = 1 does column binding (same row number).
        labels_overlap_stoich = {}
        for i,channel in enumerate(self.analysis_channels):
            ## PROBLEM: at some point, self.tables['102_overlap_bool'] got a column with same name as channel that is all True (like, overlaps with everything). But is not present in self.seg.tables['004_segmented_maxima']
            labels_overlap_stoich[channel] = self.seg.tables['004_segmented_maxima'][channel].loc[:,'label'][self.tables['102_overlap_bool'].sum(axis = 1-i).to_numpy() == overlap_stoich[i]].to_numpy().tolist()
        if sum(overlap_stoich) == 0:
            ### only non-overlapping, so each region in one channel is related to nothing in the other.
            ### Special case: evaluating a channel against itself.
            if self.analysis_channels[0] == self.analysis_channels[1]:
                labels_overlap_stoich[self.analysis_channels[0]] = self.seg.tables['004_segmented_maxima'][channel].loc[:,'label'].to_numpy().tolist()
            labels_overlap_stoich_len = [len(labels_overlap_stoich[self.analysis_channels[0]]),len(labels_overlap_stoich[self.analysis_channels[1]])]
            labels_overlap_stoich[self.analysis_channels[0]] = labels_overlap_stoich[self.analysis_channels[0]] + [np.NaN]*labels_overlap_stoich_len[1]
            labels_overlap_stoich[self.analysis_channels[1]] = [np.NaN]*labels_overlap_stoich_len[0] + labels_overlap_stoich[self.analysis_channels[1]]
            labels_overlap_table = pd.DataFrame(labels_overlap_stoich)
            ### Make a layer with every non-overlapping segmented region per channel (two channel only).
            if self.analysis_channels[0] != self.analysis_channels[1]:
                stoich_overlap_layer = np.zeros(self.layers['102_segmented_labeled_all_intersections'].shape)
                for index, row in labels_overlap_table.iloc[:labels_overlap_stoich_len[0],].iterrows():
                    label = row[self.analysis_channels[0]]
                    stoich_overlap_layer[self.seg.layers['004_segmented_labeled'][self.analysis_channels[0]] == label] = 1
                for index, row in labels_overlap_table.iloc[labels_overlap_stoich_len[0]:,].iterrows():
                    label = row[self.analysis_channels[1]]
                    stoich_overlap_layer[self.seg.layers['004_segmented_labeled'][self.analysis_channels[1]] == label] = 2
                self.layers['105_stoich_{}-to-{}_segmented_labeled'.format(str(overlap_stoich[0]),str(overlap_stoich[1]))] = stoich_overlap_layer.astype(int)
        elif sum(overlap_stoich) == 2:
            ### only single colocalization, so each region in one channel is related to only one region in the other and viceversa.
            ### Right now, labels_overlap_stoich contains all labels in each channel that overlap with a single signal in the other, but is not reciprocal (the single signal in the other channel may be overlapping with more than one).
            ### We could use any of the channel labels in labels_overlap_stoich and check for reciprocal single overlap.
            labels_overlap_list = []
            for channel_1_reg in labels_overlap_stoich[self.analysis_channels[0]]:
                channel_2_reg = self.tables['102_overlap_bool'].columns.to_numpy()[self.tables['102_overlap_bool'].loc[channel_1_reg,:].to_numpy()]
                if len(channel_2_reg) == 1:
                    channel_2_reg = channel_2_reg[0]
                    if sum(self.tables['102_overlap_bool'].loc[channel_1_reg,:].to_numpy()) == 1 and sum(self.tables['102_overlap_bool'].loc[:,channel_2_reg].to_numpy()) == 1:
                        labels_overlap_list.append([int(channel_1_reg),int(channel_2_reg)])
            labels_overlap_table = pd.DataFrame(labels_overlap_list, columns = self.analysis_channels)
            ### Make a layer with specific overlapping regions (channel_1 is label 1, channel_2 is label 2, overlaps have their unique label, which is a sequence of minimum two digits, making labels 1-9 useful for our own labeling).
            stoich_overlap_layer = np.zeros(self.layers['102_segmented_labeled_all_intersections'].shape)
            for index, row in labels_overlap_table.iterrows():
                stoich_overlap_layer[self.seg.layers['004_segmented_labeled'][self.analysis_channels[0]] == row[self.analysis_channels[0]]] = 1
                stoich_overlap_layer[self.seg.layers['004_segmented_labeled'][self.analysis_channels[1]] == row[self.analysis_channels[1]]] = 2
                overlap_label = row[self.analysis_channels[0]]*10**self.max_label + row[self.analysis_channels[1]]
                stoich_overlap_layer[self.layers['102_segmented_labeled_all_intersections'] == overlap_label] = overlap_label
            self.layers['105_stoich_{}-to-{}_segmented_labeled'.format(str(overlap_stoich[0]),str(overlap_stoich[1]))] = stoich_overlap_layer.astype(int)
        else:
            ### First, find which side has higher stoichiometry value, and then validate that every element in the other axis is overlap == 1.
            n_overlaps_axis = np.where(overlap_stoich > 1)[0][0]    ## A bit complicated, overlap_stoich seems to have 2 dimensions? But one is empty, so retrieve first ndarray, and from it retrieve first element to get the integer value.
            n_df_cols = overlap_stoich.tolist()[n_overlaps_axis]
            df_colnames = ['{}_{}'.format(channel,sequence) for channel,sequence in zip([self.analysis_channels[int(1-n_overlaps_axis)]]*n_df_cols,list(range(n_df_cols)))]
            df_colnames = [self.analysis_channels[int(n_overlaps_axis)]]+df_colnames
            ## Easier to apply things following same axis. If we do (N,1), rows overlap with multiple and columns with single.
            ## If we do (1,N), opposite way. So we just pivot DataFrame and always have rows overlap multiple.
            if n_overlaps_axis == 1:
                overlap_table_newrownames = self.tables['102_overlap_bool'].columns.to_list()
                overlap_table_newcolnames = self.tables['102_overlap_bool'].index.to_list()

                overlap_table = pd.DataFrame(data = self.tables['102_overlap_bool'].to_numpy().transpose(),
                                             index = overlap_table_newrownames,
                                             columns = overlap_table_newcolnames)
            else:
                overlap_table = self.tables['102_overlap_bool']
            labels_overlap_list = []
            for row_label in overlap_table.index.to_list():
                rowdata = overlap_table.loc[row_label,:]
                ## Only want rows that overlap with N elements, could be (N,1) or (1,N).
                if np.sum(rowdata) == overlap_stoich[n_overlaps_axis]:
                    colsdata = overlap_table.iloc[:,np.where(rowdata == True)[0]]
                    ## Since cols should overlap only with selected row, each column should have one overlap, the N columns should have N overlaps in total.
                    if np.sum(np.asarray(colsdata)) == overlap_stoich[n_overlaps_axis]:
                        col_labels = colsdata.columns.to_list()
                        labels_overlap_list.append([row_label,*col_labels])
            labels_overlap_table = pd.DataFrame(labels_overlap_list, columns = df_colnames)
            if n_overlaps_axis == 1:
                ## Put first column at the end, since we want to maintain self.analysis_channels order.
                first_col = labels_overlap_table.pop(labels_overlap_table.columns[0])
                labels_overlap_table[self.analysis_channels[n_overlaps_axis]] = first_col
            ### Make a layer with specific overlapping regions (channel_1 is label 1, channel_2 is label 2, overlaps have their unique label, which is a sequence of minimum two digits, making labels 1-9 useful for our own labeling).
            stoich_overlap_layer = np.zeros(self.layers['102_segmented_labeled_all_intersections'].shape)
            for index, row in labels_overlap_table.iterrows():
                # df_colnames keeps the order of single-label, multiple-labels
                # n_overlap_axis gives the channel index for channel_name that is single-label.
                # Label of single-label channel in the segmented layer would be n_overlap_axis + 1 (position 0 is 1, position 1 is 2).
                # Label of multiple-label channel in the segmented layer would b 2 - n_overlap_axis (position 1 (1-0) is 2, position 0 (1-1) is 1)
                stoich_overlap_layer[self.seg.layers['004_segmented_labeled'][self.analysis_channels[int(n_overlaps_axis)]] == row[self.analysis_channels[int(n_overlaps_axis)]]] = int(n_overlaps_axis) + 1
                for multi_channel_col in df_colnames[1:]:
                    stoich_overlap_layer[self.seg.layers['004_segmented_labeled'][self.analysis_channels[int(1-n_overlaps_axis)]] == row[multi_channel_col]] = 2 - int(n_overlaps_axis)
                    label_pair = [0,0]
                    label_pair[int(n_overlaps_axis)] = row[self.analysis_channels[int(n_overlaps_axis)]]
                    label_pair[int(1-n_overlaps_axis)] = row[multi_channel_col]
                    overlap_label = label_pair[0]*10**self.max_label + label_pair[1]
                    stoich_overlap_layer[self.layers['102_segmented_labeled_all_intersections'] == overlap_label] = overlap_label
            self.layers['105_stoich_{}-to-{}_segmented_labeled'.format(str(overlap_stoich[0]),str(overlap_stoich[1]))] = stoich_overlap_layer.astype(int)
        self.overlap_stoich = overlap_stoich
        self.tables['105_overlap_labels'] = labels_overlap_table
        self.time_log['simple_overlap'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, simple overlap detection {}-to-{} ({} vs {}): {}'.format(str(overlap_stoich[0]),str(overlap_stoich[1]),self.analysis_channels[0],self.analysis_channels[1],self.time_log['simple_overlap']))
        self.start_time = time.time()


    def summarize_vector_metrics(self,vector, pixel_unit_conversion):
        '''
        vector cannot be pd.Series, should be only the values
        '''
        vector_conv = np.asarray(vector)*pixel_unit_conversion
        vector_summary = {}
        vector_summary['data'] = vector_conv
        vector_summary['N'] = len(vector_conv)
        vector_summary['mean'] = np.mean(vector_conv)
        vector_summary['median'] = np.median(vector_conv)
        vector_summary['std'] = np.std(vector_conv)
        vector_summary['1st_quartile'] = np.quantile(vector_conv, 0.25)
        vector_summary['3rd_quartile'] = np.quantile(vector_conv, 0.75)
        return vector_summary


    def make_summary_plots(self,dataframe_lists, pixel_unit_conversion,mode='mean',unit='micron'):
        '''
        dataframe_lists is list of dataframes containing data to be summarized and plotted. One plot row per dataframe in list.
        mode can be 'mean' to make mean+std plots, or 'median' to be median+IQR plots
        '''
        main_title = '{}-to-{} overlap analysis of {} vs {}'.format(self.overlap_stoich[0],self.overlap_stoich[1],self.analysis_channels[0],self.analysis_channels[1])
        max_ncols = max([df.shape[1] for df in dataframe_lists])
        base_size = 8
        plt.figure(figsize = (max_ncols*base_size,len(dataframe_lists)*base_size))
        if mode == 'mean':
            main_title += ' (mean -+ std)'
        elif mode == 'median':
            main_title += ' (median with IQR)'
        plt.suptitle(main_title, fontsize=60)
        for row,df in enumerate(dataframe_lists):
            plot_n = row*max_ncols
            if df.columns[0][3:] in self.analysis_channels:
                plot_color = self.seg.channel_colors[df.columns[0][3:]]
            else:
                plot_color = 'gray'
            for col_number in range(1,df.shape[1]):
                pixel_conv = copy.copy(pixel_unit_conversion)   #Recover original value at the start of every loop, make sure it does not change original value
                plot_n += 1
                col_key = df.columns[col_number]
                if col_key == 'EMPTY':
                    continue
                col_values = df.loc[:,col_key].to_numpy()
                if len(col_values) == 0:
                    plt.subplot(len(dataframe_lists),max_ncols,plot_n)
                    plt.xlabel('{} {} (N = {})'.format(df.columns[0][3:],col_key,str(df.shape[0])))
                    continue
                if col_key.startswith('Volume'):
                    ## Now a pixel could be considered a cube, or edge of a cube
                    ## If it is a cube, pixel_conv**3
                    ## If it is edge of a pixel_conv*pixel_conv*pixel_conv, then it is an eighth of pixel_conv**3
                    pixel_conv = (pixel_conv**3)
                max_value_plot = max(np.asarray(col_values)*pixel_conv)
                binwidth = 10**(np.floor(np.log10(max_value_plot))-1)
                binrange = [0,max_value_plot]
                if col_key.startswith('Volume'):
                    if unit == 'mili':
                        unit_plot = 'mm$^3$'
                    elif unit == 'micron':
                        unit_plot = '\u03bcm$^3$'
                    elif unit == 'nano':
                        unit_plot = 'nm$^3$'
                    elif unit == 'pixel':
                        unit_plot = 'voxels'
                elif col_key.startswith('Ratio'):
                    binwidth = 0.05
                    binrange = [0,1]
                    unit_plot = ''
                    pixel_conv = 1
                else:
                    if unit == 'mili':
                        unit_plot = 'mm'
                    elif unit == 'micron':
                        unit_plot = '\u03bcm'
                    elif unit == 'nano':
                        unit_plot = 'nm'
                    elif unit == 'pixel':
                        unit_plot = 'pixel'
                col_summary = self.summarize_vector_metrics(col_values, pixel_unit_conversion=pixel_conv)
                plt.subplot(len(dataframe_lists),max_ncols,plot_n)
                ax = sns.histplot(data=col_summary['data'], color=plot_color, kde=True, binwidth = binwidth, binrange = binrange)
                try:
                    ax.lines[0].set_color('black')
                except:
                    pass
                plt.xlabel('{} {} (N = {})'.format(df.columns[0][3:],col_key,str(df.shape[0])))
                plt.xticks(rotation=45)
                if mode == 'mean':
                    txt = 'mean \u00B1 std: {}{} \u00B1 {}{}'.format(str(round(col_summary['mean'],4)),unit_plot,str(round(col_summary['std'],4)),unit_plot)
                elif mode == 'median':
                    txt = 'median [IQR]: {} {} [{}{} - {}{}]'.format(str(round(col_summary['median'],4)),unit_plot,str(round(col_summary['1st_quartile'],4)),unit_plot,str(round(col_summary['3rd_quartile'],4)),unit_plot)
                txt += '\n(min - max: {}{} - {}{})'.format(str(round(min(col_summary['data']),4)),unit_plot,str(round(max(col_summary['data']),4)),unit_plot)
                plt.plot([],[], ' ', label=txt)
                plt.legend()
        hist_path = '{}/{}_{}_stoich{}_hist_{}_{}.png'.format(self.seg.RES_IMAGE_PATH,self.analysis_channels[0],self.analysis_channels[1],str(self.overlap_stoich),mode,unit)
        plt.savefig(hist_path)
        plt.close()


    def measure_overlapping_regions(self, pixel_unit_conversion = 0.025, unit = 'micron'):
        '''
        Extract relevant metrics from overlapping regions identified by 'find_simple_overlap_regions'.

        Identifier: 106

        Uses    self.seg.tables['004_segmented_regions'] -> segmented regions information (dictionary, each entry key is a channel). 'label' column information contains unique identifier for that channel.
                self.seg.tables['004_segmented_maxima']
                self.tables['102_intersection_regions'] -> overlap-specific information, two initial columns contain the regions involved. 'label' column contains unique identifier as label_1*10**max_label+label_2.
                self.tables['103_neighbor_centroid_to_centroid_dist']
                self.tables['105_overlap_labels] -> each row, combinations of labels in each channel that satisfy the overlap restrictions.
        '''
        ### There are 3 potential cases per row:
        # One column has integer, the other column has np.NaN.
        # Both columns have integers
        # More than two columns.
        if sum(self.tables['105_overlap_labels'].isna().sum()) > 0: ## Original idea was using any(self.tables['105_overlap_labels'].isna()). However, it was True even if there were no NAs.
            # Non-overlapping regions. Report only volume?
            non_overlap_labels = {}
            volume_each = {}
            for channel in self.analysis_channels:
                non_overlap_labels[channel] = self.tables['105_overlap_labels'].loc[:,channel]
                non_overlap_labels[channel] = non_overlap_labels[channel][np.invert(pd.isnull(non_overlap_labels[channel]))].to_numpy()
                volume_each[channel] = np.empty((0))
                for label in non_overlap_labels[channel]:
                    volume_each[channel] = np.concatenate([volume_each[channel],self.seg.tables['004_segmented_regions'][channel].loc[self.seg.tables['004_segmented_regions'][channel]['label'] == label,'area'].to_numpy()])
            df_channels = {}
            for channel in self.analysis_channels:
                df_channels[channel] = pd.DataFrame(list(zip(non_overlap_labels[channel], volume_each[channel])), columns = ['ID_{}'.format(channel),'Volume'])
            plot_dfs = [df_channels[key] for key in df_channels]
        #
        ##
        ###
        ####
        #####
        elif self.tables['105_overlap_labels'].shape[1]>2:
            # Have more than 2 columns, because from we retrieved more than one region per channel (could be one channel or both).
            # Order values per channel if more than one so smaller volume goes first?
            # Report volumes, overlap volumes, ratio of overlap volume vs bigger volumes, distances to centers, angles (skeletonized version?).
            # Results have to go in a list of DataFrames called plot_dfs
            channel_per_col = [self.analysis_channels[0]]*self.overlap_stoich[1]+[self.analysis_channels[1]]*self.overlap_stoich[0]
            n_overlaps_channel = self.analysis_channels[np.where(self.overlap_stoich>1)[0][0]]
            label_channels = {}                         # one per label
            label_intersections = {}                    # one per intersection, which is N-1 in our case
            volume_each = {}                            # one per label
            volume_overlaps = {}                        # N-1
            volume_combined = np.empty(0)               # only one, total thing
            distance_centroid_to_intersection = {}      # two per intersection, so 2N-2
            distance_maxima_to_intersection = {}        # 2N-2
            distance_between_centroids = np.empty(0)    # one per intersection. In our case, N-1
            distance_between_maxima = np.empty(0)       # N-1
            angle_between_overlaps = np.empty(0)        # two intersections have 1 only
            angle_between_nreg_centroids = np.empty()   # two intersections have 1 only
            angle_between_nreg_maxima = np.empty()   # two intersections have 1 only
            ratio_each = {}                             # N-1
            ratio_overlap = np.empty(0)                 # N-1
            for col in self.tables['105_overlap_labels'].columns:
                label_channels[channel] = np.empty(0)
                volume_each[channel] = np.empty(0)
                ratio_each[channel] = np.empty(0)
                distance_centroid_to_intersection[channel] = np.empty(0)
                distance_maxima_to_intersection[channel] = np.empty(0)
            for index,intersection_row in self.tables['105_overlap_labels'].iterrows():
                pass
        else:
            # Assume each is an integer, although it could be other things that we are not checking.
            # Report volumes, overlap volume, ratio of overlap volume vs bigger volumes, distance between centers.
            label_channels = {}
            label_intersection = np.empty(0)
            volume_each = {}
            volume_overlap = np.empty(0)
            volume_combined = np.empty(0)
            distance_centroid_to_intersection = {}
            distance_maxima_to_intersection = {}
            distance_between_centroids = np.empty(0)
            distance_between_maxima = np.empty(0)
            ratio_each = {}
            ratio_overlap = np.empty(0)
            for channel in self.analysis_channels:
                label_channels[channel] = np.empty(0)
                volume_each[channel] = np.empty(0)
                ratio_each[channel] = np.empty(0)
                distance_centroid_to_intersection[channel] = np.empty(0)
                distance_maxima_to_intersection[channel] = np.empty(0)
            for index,intersection_row in self.tables['105_overlap_labels'].iterrows():
                label_channel_1 = intersection_row[self.analysis_channels[0]]
                label_channel_2 = intersection_row[self.analysis_channels[1]]
                region_channels = {}
                center_channels = {}
                for channel in self.analysis_channels:
                    region_row = self.seg.tables['004_segmented_regions'][channel]['label'] == intersection_row[channel]
                    region_channels[channel] = self.seg.tables['004_segmented_regions'][channel].loc[region_row,:]
                    center_row = self.seg.tables['004_segmented_maxima'][channel]['label'] == intersection_row[channel]
                    center_channels[channel] = self.seg.tables['004_segmented_maxima'][channel].loc[center_row,:]
                region_intersection = (self.tables['102_intersection_regions'][self.analysis_channels[0]].to_numpy() == label_channel_1)*(self.tables['102_intersection_regions'][self.analysis_channels[1]].to_numpy() == label_channel_2)
                region_intersection = self.tables['102_intersection_regions'].loc[region_intersection,:]
                #### USE NUMPY CONCATENATE INSTEAD OF LIST APPEND
                label_intersection = np.concatenate([label_intersection, region_intersection['label'].to_numpy()])
                volume_overlap = np.concatenate([volume_overlap, region_intersection['area'].to_numpy()])
                volume_combined = np.concatenate([volume_combined, region_channels[self.analysis_channels[0]]['area'].to_numpy()+region_channels[self.analysis_channels[1]]['area'].to_numpy()-volume_overlap[-1]])
                ratio_overlap = np.concatenate([ratio_overlap, np.array([volume_overlap[-1]/volume_combined[-1]])])
                if '103_neighbor_centroid_to_centroid_dist' not in list(self.tables.keys()):
                    self.neighbor_distance()
                distance_between_centroids = np.concatenate([distance_between_centroids, np.array([self.tables['103_neighbor_centroid_to_centroid_dist'].loc[label_channel_1,label_channel_2]])])
                distance_between_maxima = np.concatenate([distance_between_maxima, np.array([self.tables['103_neighbor_centroid_to_centroid_dist'].loc[label_channel_1,label_channel_2]])])
                intersection_centroid = region_intersection.loc[:,['centroid-0','centroid-1','centroid-2']]
                for channel in self.analysis_channels:
                    label_channels[channel] = np.concatenate([label_channels[channel], region_channels[channel]['label'].to_numpy()])
                    volume_each[channel] = np.concatenate([volume_each[channel], region_channels[channel]['area'].to_numpy()])
                    ratio_each[channel] = np.concatenate([ratio_each[channel], np.array([volume_overlap[-1]/volume_each[channel][-1]])])
                    # Euclidean distance is sqrt(sum of squares of the difference in each point coordinate)
                    def euclidean_dist(vector_1,vector_2):
                        vector_1 = np.asarray(vector_1)
                        vector_2 = np.asarray(vector_2)
                        if vector_1.shape == vector_2.shape:
                            diff = vector_1.flatten() - vector_2.flatten()
                            square_diff = np.power(diff,2)
                            sum_square_diff = sum(square_diff)
                            dist = np.sqrt(sum_square_diff)
                            return dist
                        else:
                            raise Exception('vectors for euclidean distance do not have the same length')
                    channel_center = center_channels[channel].loc[:,['centroid_z','centroid_y','centroid_x']]
                    distance_centroid_to_intersection[channel] = np.concatenate([distance_centroid_to_intersection[channel], np.array([euclidean_dist(intersection_centroid,channel_center)])])
                    channel_center = center_channels[channel].loc[:,['maxima_z','maxima_y','maxima_x']]
                    distance_maxima_to_intersection[channel] = np.concatenate([distance_maxima_to_intersection[channel], np.array([euclidean_dist(intersection_centroid,channel_center)])])
            df_channels = {}
            for channel in self.analysis_channels:
                df_channels[channel] = pd.DataFrame(zip(label_channels[channel], volume_each[channel], np.zeros(len(label_channels[channel])), ratio_each[channel],
                                                        distance_centroid_to_intersection[channel], distance_maxima_to_intersection[channel]),
                        columns = ['ID_{}'.format(channel),'Volume', 'EMPTY', 'Ratio of overlap vs channel',
                                   'Distance from channel centroid to intersection centroid','Distance from intensity maxima to intersection centroid'])
            df_intersection = pd.DataFrame(zip(label_intersection, volume_overlap, volume_combined, ratio_overlap, distance_between_centroids, distance_between_maxima),
                    columns = ['ID_{}_vs_{}'.format(*self.analysis_channels), 'Volume of overlap','Volume total','Ratio of overlap vs total',
                               'Distance between channel centroid','Distance between channel intensity maxima'])
            plot_dfs = [df_channels[key] for key in df_channels]
            plot_dfs.insert(0, df_intersection)
        self.make_summary_plots(plot_dfs,pixel_unit_conversion=pixel_unit_conversion,mode='mean',unit=unit)
        self.make_summary_plots(plot_dfs,pixel_unit_conversion=pixel_unit_conversion,mode='median',unit=unit)
        self.tables['106_channel_measurements'] = df_channels
        if sum(self.overlap_stoich) != 0:
            self.tables['106_overlap_measurements'] = df_intersection
        self.time_log['measure_overlap'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, measuring overlap properties and making summary plots {}-to-{} ({} vs {}): {}'.format(str(self.overlap_stoich[0]),str(self.overlap_stoich[1]),*self.analysis_channels,self.time_log['measure_overlap']))
        self.start_time = time.time()


    def single_channel_cleanup(self):
        '''
        Remove layers and tables not relevant when comparing a channel with itself, because:
        - Information is same as SIM_segmentation output
        - Redundant (example, overlap is 100%)
        '''
        if self.analysis_channels[0] == self.analysis_channels[1]:
            del self.layers['102_union_mask']
            del self.layers['102_segmented_labeled_all_intersections']
            del self.tables['102_overlap_bool']
            del self.tables['102_intersection_regions']
            del self.tables['105_overlap_labels']


    def save_segmentation_layers(self, prefix = '', suffix = '', common_results = True):
        '''
        Same as in SIM_segmentation object class.

        common_results: boolean (default, True). If True, will return layers and tables always computed (all intersections, all distances) together with stoichiometry-specific results.
        '''
        save_dir = os.path.join(self.seg.RES_IMAGE_PATH,'overlap_layers')
        check_to_make_dir(save_dir)
        for layer_key in self.layers.keys():
            if not common_results:
                if any(['102' in layer_key, '103' in layer_key, '104' in layer_key]):
                    continue
            if type(self.layers[layer_key]) == dict:
                for channel in self.analysis_channels:
                    f = gzip.GzipFile('{}/{}{}_{}{}_{}.npy.gz'.format(save_dir, prefix, '{}_vs_{}'.format(*self.analysis_channels), layer_key, suffix, channel), "w")
                    np.save(file=f, arr=self.layers[layer_key][channel])
                    f.close()
            else:
                f = gzip.GzipFile('{}/{}{}_{}{}.npy.gz'.format(save_dir,prefix,'{}_vs_{}'.format(*self.analysis_channels),layer_key,suffix), "w")
                np.save(file=f, arr=self.layers[layer_key])
                f.close()
        save_dir = os.path.join(self.seg.RES_IMAGE_PATH,'overlap_tables')
        check_to_make_dir(save_dir)
        for table_key in self.tables.keys():
            if not common_results:
                if any(['102' in table_key, '103' in table_key, '104' in table_key]):
                    continue
            if type(self.tables[table_key]) == dict:
                for channel in self.analysis_channels:
                    CSV_path = '{}/{}{}_{}{}_{}.csv.gz'.format(save_dir,prefix,'{}_vs_{}'.format(*self.analysis_channels),table_key,suffix,channel)
                    if not os.path.exists(CSV_path):
                        self.tables[table_key][channel].to_csv(CSV_path, sep = '\t', na_rep = 'NaN',index=True, index_label = 'INDEX', compression = 'gzip')
            else:
                CSV_path = '{}/{}{}_{}{}.csv.gz'.format(save_dir,prefix,'{}_vs_{}'.format(*self.analysis_channels),table_key,suffix)
                if not os.path.exists(CSV_path):
                    self.tables[table_key].to_csv(CSV_path, sep = '\t', na_rep = 'NaN',index=True, index_label = 'INDEX', compression = 'gzip')
        self.time_log['save_quant'] = str(round(time.time() - self.start_time,4))
        print('Seconds elapsed, saving quantified overlap analysis layers and tables: {}'.format(self.time_log['save_quant']))
        self.start_time = time.time()





    def legacy(self, k, d):
        '''
        IGNORE FOR FUNCTIONAL PURPOSES.
        Work that took a long time to make, but has lost its purpose.
        '''
        center = [center+i for i in['_z','_y','_x']]
        dists = pairwise_distances(np.asarray(self.seg.tables['004_segmented_maxima'][self.channel_1].loc[:,center]), np.asarray(self.seg.tables['004_segmented_maxima'][self.channel_2].loc[:,center]))
        labels_1 = self.seg.tables['004_segmented_maxima'][self.channel_1].loc[:,'label']
        labels_2 = self.seg.tables['004_segmented_maxima'][self.channel_2].loc[:,'label']
        labels_and_linked_labels_dists = {}
        for label_1,dists_to_label in zip(labels_1,dists):
            sorted_dists_labels = sorted(zip(dists_to_label,labels_2))
            # In here, we can easily sort from lowest to highest, but we also want the label identity. By zipping the lists, every value pair is linked by making tuples, but only first element of each tuple is used for ordering.
            if k > 0:
                sorted_dists_labels = sorted_dists_labels[:k]
            if d > 0:
                sorted_dists = list(zip(*sorted_dists_labels))[0]
                # This is to extract list of sorted distances from the tuple pairs generated before. Zip always takes a group of lists of same length and generates n-tuples of i elements, n being lists length and i number of lists.
                # The * right before the list of tuples indicates to get inside the list, so it returns every element inside the list (they are not inside the list anymore, but treated like independent lists).
                # Since * returns all tuples as independent lists, it can be given to zip and return every i-th inside the n tuples separate as i lists of n length.
                # Probably already realized by now, but short summary. list(zip(list_1, list_2, ... list_n)) returns a list of i tuples of n elements per tuple.
                # list(zip(*result_of_zipped_lists)) returns n lists of i elements each (could be original lists, unless you do stuff like sorting in this case).
                #### WE DO BASICALLY 2D NUMPY ARRAY TRANSPOSITION USING BASE PYTHON
                if d in sorted_dists:
                    sorted_dists_labels = sorted_dists_labels[:bisect.bisect_right(sorted_dists, d)]
                else:
                    sorted_dists_labels = sorted_dists_labels[:bisect.bisect_left(sorted_dists, d)]
                # bisect_left evaluates where is the right-most element in the list lower than the value indicated, and returns index+1  (so you could immediately slice or insert)
                # if element is in list, bisect_left returns the index when encountering that value for the first time (so you could insert or slice right before it).
                # In case of including that element (do <= instead of <), do bisect_right so it returns index+1 for last time the element found in sorted list.
            labels_and_linked_labels_dists[label_1] = sorted_dists_labels
        self.neighbors = labels_and_linked_labels_dists
        df = pd.DataFrame(dists, columns = np.asarray(self.seg.tables['004_segmented_maxima'][self.channel_2].loc[:,'label']), index = np.asarray(self.seg.tables['004_segmented_maxima'][self.channel_1].loc[:,'label']))
        # When making the dataframe, columns are for channel_2 and rows are for channel_1. This comes from how matrices are transformed into DataFrames (each column is made of each i-th element in every innermost list).
        