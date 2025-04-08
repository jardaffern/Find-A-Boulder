'''
For a given set of hillshades and boulder information extract the relevant overlaps

1) Pick a 'zoom'
2) Go to where there are boulders
3) Extract the information and the TIF that has info for the boulders

Note that: some boulders will not have a DEM or hillshade available. We save all
boudlers previously and in this script we subset to boulders we were 'successful'
with

'''

import pyproj
from osgeo import gdal, osr
import geopandas as gpd
import numpy as np
import pickle
from pyproj import Transformer

from os import listdir
from os.path import isfile, join


def rolling_window(arr: np.ndarray, window_size: tuple = (3, 3)) -> np.ndarray:
    
    """
    Gets a view with a window of a specific size for each element in arr.

    Parameters
    ----------
    arr : np.ndarray
        NumPy 2D array.
    window_size : tuple
        Tuple with the number of rows and columns for the window. Both values
        have to be positive (i.e. greater than zero) and they cannot exceed
        arr dimensions.

    Returns
    -------
    NumPy 4D array

    Notes
    -----
    This function has been slightly adapted from the one presented on:
    https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html.

    It is advised to read the notes on the numpy.lib.stride_tricks.as_strided
    function, which can be found on:
    https://docs.scipy.org/doc/numpy-1.17.0/reference/generated/numpy.lib.stride_tricks.as_strided.html
    """

    # validate window size
    err1 = 'window size must be postive'
    err2 = 'window size exceeds input array dimensions'
    assert window_size[0] > 0 and window_size[1] > 0, err1
    assert window_size[0] <= arr.shape[0] and window_size[1] <= arr.shape[1], err2

    # calculate output array's shape
    y_size = (arr.shape[0] - window_size[0]) + 1
    x_size = (arr.shape[1] - window_size[1]) + 1
    shape = (y_size, x_size) + window_size

    # define strides
    strides = arr.strides * 2

    return np.lib.stride_tricks.as_strided(arr, shape, strides, writeable=False)


def get_indices(x: np.ndarray, y: np.ndarray, ox: float, oy: float,
                pw=1.0, ph=1.0) -> tuple:
    
    """
    Gets the row (i) and column (j) indices in an NumPy 2D array for a given
    set of coordinates.

    Parameters
    ----------
    x : np.ndarray
        NumPy 1D array containing the x (longitude) coordinates.
    y : np.ndarray
        NumPy 1D array containing the y (latitude) coordinates.
    ox : float
        Raster x origin (minimum x coordinate)
    oy : float
        Raster y origin (maximum y coordinate)
    pw : float
        Raster pixel width
    ph : float
        Raster pixel height

    Returns
    -------
    Two-element tuple with the column and row indices.

    Notes
    -----
    This function is based on: https://gis.stackexchange.com/a/92015/86131.

    All x and y coordinates must be within the raster boundaries. Otherwise,
    indices will not correspond to the actual values or will be out of bounds.
    """
    # make sure pixel height is positive
    ph = abs(ph)

    i = np.floor((oy-y) / ph).astype('int')
    j = np.floor((x-ox) / pw).astype('int')

    return i, j

def gdal_to_lat_long(hillshade) -> None:

    'Convert coordiantes to something else!'

    InSR = osr.SpatialReference()
    InSR.ImportFromEPSG(4326)  

    hillshade.SetProjection(InSR.ExportToWkt())
    
    return hillshade


def modify_row_crs(new_crs,
                   geo_data):

    current_crs = geo_data.crs.to_epsg()

    trans = Transformer.from_crs(
        f"EPSG:{current_crs}",
        f"EPSG:{new_crs}",
        always_xy=True,
    )
    xx, yy = trans.transform(geo_data["x"].values, geo_data["y"].values)

    return xx, yy

def calculate_modified_crs(boulders:gpd.GeoDataFrame,
                           hill_geo:str) -> gpd.GeoDataFrame:

    modified_coords = []
    for id in boulders['unique-id']:

        boulder = boulders[boulders['unique-id'] == id]

        modified_x, modified_y = modify_row_crs(new_crs=hill_geo, 
                                                geo_data=boulder)

        modified_coords.append({'new_x':modified_x,'new_y':modified_y})

    new_x_name = hill_geo + '_x'
    new_y_name = hill_geo + '_y'

    boulders[new_x_name] = [x['new_x'][0] for x in modified_coords]        
    boulders[new_y_name] = [x['new_y'][0] for x in modified_coords]

    return boulders

def create_mask(boulder_index: np.array,
                            image_size = (640,640),
                            boulder_size = (7,7)) -> np.array:

    '''
    Provide a specific index and return a masked array that contains
    all the indices for which that boulder was found
    '''

    mask_array = np.zeros(shape = image_size)

    indices = []

    for boulder in boulder_index:

        x_min = boulder[0] - boulder_size[0]
        x_max = boulder[0] + boulder_size[0]

        y_min = boulder[1] - boulder_size[1]
        y_max = boulder[1] + boulder_size[1]

        mask_array[x_min:x_max,
                   y_min:y_max] = 1

        indices.append({'x_min':x_min,
                        'x_max':x_max,
                        'y_min':y_min,
                        'y_max':y_max})

    return mask_array, indices


def match_boulder_to_hillshade(hillshade_data:list,
                               hillshade_list:list,
                               boulders:gpd.GeoDataFrame) -> dict:
 
    '''
    For one particular row (boulder) match the .tif that contains it
    '''

    boulder_values=[]
    
    for hill_index, hill in enumerate(hillshade_data):

        print(f'Working on {hillshade_list[hill_index]}')
        #Check if point is in the boundary!
        ox, pw, _, oy, _, ph = hill.GetGeoTransform()
        lrx = ox + (hill.RasterXSize * pw)
        lry = oy + (hill.RasterYSize * ph)
        nd_value = hill.GetRasterBand(1).GetNoDataValue()
        arr = hill.ReadAsArray()

        #Convert x y to right UTM?
        hill_geo = osr.SpatialReference(wkt=hill.GetProjection()).GetAttrValue('AUTHORITY',1)

        x_coord_col = hill_geo + '_x'
        y_coord_col = hill_geo + '_y'

        to_sub = boulders[(boulders[x_coord_col] > ox) &
                          (boulders[x_coord_col] < lrx) &
                          (boulders[y_coord_col] < oy) & 
                          (boulders[y_coord_col] > lry)]

        #If the point is not in the raster then keep looking
        if len(to_sub) == 0:
            continue

        #Now reproject for the get_indices?
        hill_lat = gdal.Open(HILLSHADES+hillshade_list[hill_index])
        gdal_to_lat_long(hill_lat)

        #GDAL/numpy version - jal hand checked that these seem OK
        window_size = (5,5)  # 16 cells incl  center
        padding_y = (2,2)  # 2 rows above and 2 rows below
        padding_X = (2,2)  # 2 columns to the left and 2 columns to the right
        padded_arr = np.pad(arr, pad_width=(padding_y, padding_X), mode='constant',
                             constant_values=nd_value)
        windows = rolling_window(padded_arr, window_size=window_size)

        boulder_matches = []
        boulder_hill_values = []
        for id in to_sub['unique-id']:

            one_boulder = to_sub[to_sub['unique-id'] == id]
            
            x_val = one_boulder[x_coord_col].to_list()[0]
            y_val = one_boulder[y_coord_col].to_list()[0]
            
            idx = get_indices(x_val, y_val, ox, oy, pw, ph)
            values = windows[idx]

            boulder_matches.append(idx)
            boulder_hill_values.append(values)

        mask, indices = create_mask(boulder_index=boulder_matches)

        matched_boulders = {'hillshade':hillshade_list[hill_index],
                            'hill_values':boulder_hill_values,
                            'boulder_info':to_sub,
                            'mask':mask,
                            'boulder_indices': indices}

        boulder_values.append(matched_boulders)

    return boulder_values

#To view some of these results manually...
# import rasterio
# raster = rasterio.open('C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/tiled/0_angeles-national-forest_fall_2023_hillshade_tile_300_1200.tif')
# to_output = raster.read(1)
# keyword_args = raster.meta
# keyword_args.update(dtype=rasterio.uint8,
#                     count=2,
#                     compress='lzw')

# final_output = np.zeros(to_output.shape, dtype=rasterio.uint8)
# final_output = final_output + to_output

# with rasterio.open('C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/test.tif', 'w', **keyword_args) as dt:
#     dt.write_band(1, final_output.astype(rasterio.uint8))
#     dt.write_band(2, matched_boulders['mask'].astype(rasterio.uint8))


if __name__ == '__main__':

    HILLSHADES = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/tiled/yolo/'
    BOULDERS = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/training_data/'

    #Verify that the geo is lat long!!
    boulders = gpd.read_file(BOULDERS+'hq_training_data_utm.shp')

    #Read in the hillsahdes
    hillshade_list = [x for x in listdir(HILLSHADES) if isfile(join(HILLSHADES, x))]
   
    hillshade_data = [gdal.Open(HILLSHADES+x) for x in hillshade_list]

    unique_projections = [osr.SpatialReference(wkt=x.GetProjection()).GetAttrValue('AUTHORITY',1) for x in hillshade_data]
    
    #Now get the unique unique projections
    unique_projections = list(set(unique_projections))

    #We add a column for each unique projection. Reprojecting takes a second
    #so we try and do this once by adding n number of lat/long type columns
    #This is due to us using UTMs
    for proj in unique_projections:

        boulders = calculate_modified_crs(boulders=boulders,
                                            hill_geo=proj)

   
    tif_tiled_results = match_boulder_to_hillshade(hillshade_data=hillshade_data,
                                                   hillshade_list=hillshade_list,
                                                   boulders=boulders)

    ##NOTE:
    #tif_tiled_results should have 2501 boulders in it. The mask stores
    #the 0:absence, 1: presence of aboulder
    #the vast majority of .tif tiled files should not be present in the
    #list.
    
    ##NOTE: There are some section that are completely black. It's hard to tell 
    #if these are DEM errors or how ia mtiling data. Will need to look for 
    #instances where i am labeling a completely black square as with a boulder.
    with open('C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/labeled_data/tif_boulders_with_mask_yolo.pkl', 'wb') as pkl_file:
        pickle.dump(tif_tiled_results, pkl_file)
