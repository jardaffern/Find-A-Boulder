'''
QAQC steps: 
    1) Check to see if all points overlap DEMs
    2) dissolve all the dems so there aren't overlapping DEMs
'''

import geopandas as gpd
from os import listdir
from os.path import isfile, join
import pandas as pd

if __name__ == '__main__':

    SHAPEFILE_DIR = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/training_data/'
    DEM_DIR = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/dem/'

    boulders = gpd.GeoDataFrame.from_file(f'{SHAPEFILE_DIR}all_training_data.shp')

    dem_files = [x for x in listdir(DEM_DIR) if isfile(join(DEM_DIR, x))]

    boulder_list = []
    for dem in dem_files:

        dem_data = gpd.GeoDataFrame.from_file(DEM_DIR+dem)

        boulders_in_dem = gpd.sjoin(boulders, dem_data, op='within')

        #Convert to datframe - dont care if its spatial anymore
        boulder_list.append(pd.DataFrame(boulders_in_dem))

    all_overlaps = pd.concat(boulder_list)

    overlap_no_dupl = all_overlaps.drop_duplicates()