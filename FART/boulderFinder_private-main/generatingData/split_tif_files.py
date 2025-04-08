'''
This script splits .tif files into 300x300 chunks.


'''

import pyproj
from osgeo import gdal
from pyproj import Transformer
from os import listdir, system
from os.path import isfile, join


def create_tiles(xsize, ysize,
                 input_filename,
                 out_path,
                 in_path,
                 tile_size = 640,
                 output_filename = 'tile_'):


    for i in range(0, xsize, tile_size):
        for j in range(0, ysize, tile_size):

            input_file = input_filename.replace('.tif','')

            output_name = str(out_path) + input_file + '_' + str(output_filename) + str(i) + \
                "_" + str(j)

            com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + \
                ", " + str(tile_size) + ", " + str(tile_size) + " " + str(in_path) + \
                str(input_filename) + " " + output_name + ".tif"
            system(com_string)

if __name__ == '__main__':

    HILLSHADES = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/'
    TILED_TIF_OUTPATH = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/tiled/yolo/'

    #Read in the hillsahdes
    hillshade_list = [x for x in listdir(HILLSHADES) if isfile(join(HILLSHADES, x))]
   
    hillshade_data = [gdal.Open(HILLSHADES+x) for x in hillshade_list]

    for dem, file_name in zip(hillshade_data,hillshade_list):

        width = dem.RasterXSize
        height = dem.RasterYSize

        create_tiles(xsize=width,
                     ysize=height,
                     out_path=TILED_TIF_OUTPATH,
                     in_path=HILLSHADES,
                     input_filename=file_name)


