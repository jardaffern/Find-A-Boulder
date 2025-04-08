'''
This file can probably be deleted. It was attempting to

-split up the .tif files in 300x300 sizes
-create a mask based on overlapping boulders for training data

'''

import pyproj
from osgeo import gdal, osr
import pandas as pd
import pickle

if __name__ == '__main__':

    DATA = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/labeled_data/'
    
    with open(f'{DATA}tif_boulders_with_mask_yolo.pkl','rb') as pkl_file:
        boulders_hillshade = pickle.load(pkl_file)

    hillshades = [x['hillshade'] for x in boulders_hillshade]

    masks = [x['mask'] for x in boulders_hillshade]

    boulder_indices = [x['boulder_indices'] for x in boulders_hillshade]

    labels = []
    #convert to a dataframe in each element
    for index, item in enumerate(boulders_hillshade):

        indices = item['boulder_indices']

        indices_df = pd.DataFrame(indices)
        indices_df['source'] = item['hillshade']

        labels.append(indices_df)

    
    labels_concat = pd.concat(labels)

    labels_concat['label'] = 'Boulder'

    labels_concat.to_pickle(path=DATA + 'boulders_dem_tiled_yolo.pkl')