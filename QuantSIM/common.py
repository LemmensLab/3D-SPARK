import os
import numpy as np
import pandas as pd

def check_to_make_dir(path, known_path = ''):
    '''
    Function. Evaluates if path to directory exists.
    If it does not exist, creates it.
    
    Arguments:
        path: string. Absolute or relative path.
        known_path: string. Absolute or relative path that already exists. The function will skip common root directories of the path.
    '''
    if os.path.exists(path):
        pass
        #print('Path indicated already exists: ' + path)
    else:
        path = os.path.abspath(path)
        if known_path != '':
            if not os.path.exists(known_path):
                raise Exception('Reference path does not exist.')
            known_path = os.path.abspath(known_path)
            path_levels = [path.split('/'), known_path.split('/'),[]]
            path_divergence = False
            for i in range(len(path_levels[0])):
                if i < len(path_levels[1]):
                    if path_levels[0][i] != path_levels[1][i] or i == len(path_levels[1])-1:
                        if path_divergence == False:
                            path_divergence = True
                            path_levels[2] = ['/'.join(path_levels[2])]
                path_levels[2].append(path_levels[0][i])
            path_levels = path_levels[2]
        else:
            path_levels = path.split('/')
        new_path = path_levels[0]
        for level in path_levels[1:]:
            new_path = '/'.join([new_path, level])
            if not os.path.exists(new_path):
                os.mkdir(new_path)
        print('Path created: ' + new_path)

def find_key_from_pattern(object, pattern, dict_name = '__dict__'):
    '''
    Function called by other class functions.
    From a dictionary in SIM_segmentation, retrieve single key name that matches a given pattern.

    Arguments:
    - object: Python object. Object that contains key to be retrieved. Can be its attribute names, or a dictionary inside the object.
    - pattern: string. Pattern of the key to be found in a dictionary.
    - dict_name: string. If '__dict__', checks for key in SIM_segmentation internal variables. If anything else, checks for key in dictionary inside SIM_segmentation object.
    '''
    if dict_name == '__dict__':
        dict_keys = np.array(list(object.__dict__.keys()))
    else:
        dict_keys = np.array(list(object.__dict__[dict_name].keys()))
    key_index = np.flatnonzero(np.core.defchararray.find(dict_keys,pattern)!=-1)
    if len(key_index) != 1:
        print(dict_keys)
        print(pattern)
        if len(key_index) == 0:
            raise Exception('keyword provided does not match any segmentation performed')
        elif len(key_index) > 1:
            for key_ind in key_index:
                print(dict_keys[key_ind])
            raise Exception('keyword provided is not specific enough, multiple entries retrieved')
    return dict_keys[key_index][0]

def update_labels_maxima(object, channel, keyword):
    '''
    Function called by other class functions.
    Adjust local maxima labels from initial segmentation (relevant if some regions are filtered out/change label).

    Arguments:
    - object: SIM_segmentation object.
    - channel: string. Name of channel to update labels.
    - keyword: string. Unique string to retrieve specific segmentation results for that channel to be updated.
            Keyword must direct to one entry in 'layers' and one entry in 'tables' dictionaries.
    '''
    segmented_key = find_key_from_pattern(object,keyword,'layers')
    maxima_key = find_key_from_pattern(object,keyword+'_maxima','tables')
    object.tables[maxima_key][channel].loc[:,'label'] = [object.layers[segmented_key][channel][int(round(row['maxima_z'],0))][int(round(row['maxima_y'],0))][int(round(row['maxima_x'],0))] for i,row in object.tables[maxima_key][channel].iterrows()]
    object.tables[maxima_key][channel] = object.tables[maxima_key][channel].iloc[object.tables[maxima_key][channel].loc[:,'label'].values != 0,:]