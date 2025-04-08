'''
After downloading DEMs into .tif files, convert all to hillshades

Hillshade calculations are from: https://www.neonscience.org/resources/learning-hub/tutorials/create-hillshade-py
'''

import pyproj # NOTE!!! This must be included first in order for spatial projects to be read in easily
from osgeo import gdal
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':

    DEM_DIR = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/dem/'
    OUTPUT_DIR = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/'

    dem_files = [x for x in listdir(DEM_DIR) if isfile(join(DEM_DIR, x))]

    for dem in dem_files:
        #These do have spatial proj info - UTM
        file_name = dem.replace('.tif','')

        dem_read_in = gdal.Open(DEM_DIR+dem)

        #These are outputted as UTM
        #MODIFY THIS IT LOOKS AWFUL WITH DEFAULT SETTINGS
        hillshade_output = gdal.DEMProcessing(f'{OUTPUT_DIR}{file_name}_hillshade.tif', dem_read_in, 'hillshade',
                                  altitude=45, azimuth=315)

        dem_read_in = None
        hillshade_output = None

        #Note that this produces 245 mb files instad of 64mb. Not sure if there's a good reason why?
        # hs_output = rio.open(OUTPUT_DIR+file_name+'_hillshade.tif', mode='w', driver='GTiff',
        #                      crs=rio.crs.CRS.from_string('EPSG:4326'),
        #                      height=hs_data.shape[0], width=hs_data.shape[1],
        #                      count=1, dtype=str(hs_data.dtype))
        
        # hs_output.write(hs_data, 1)
        # hs_output.close()
