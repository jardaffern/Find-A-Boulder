'''
For agiven set of coorindates, create the tiled TIFs to use for prediction

'''
import pyproj
from osgeo import gdal
import pandas as pd
from scrapingData.dem_scraper import ping_open_topo_api, get_bbox_values
from generatingData.split_tif_files import create_tiles

MODEL_LOC = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/models/'
OUTPUT_TIF_DIR = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/dem_predict/'
TILED_TIF_OUTPATH = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/dem_predict/tiled/'


if __name__ == "__main__":

    red_feather = [41.155063, -105.373943]

    to_predict = pd.DataFrame({'mcp_id':[1],
                               'lnglat':[red_feather]})

    bbox_data = get_bbox_values(mcp_data=to_predict,
                           buffer_amount=0.005)

    #Steps: 
    #look for an area to get DEM for
    #split into multiple .tif files
    #predict
    #put back together again

    #timing note: > 2 secs to get a result
    api_result = ping_open_topo_api(bbox=bbox_data.iloc[0],
                                    dataset_type='USGS1m')
    
    with open(f'{OUTPUT_TIF_DIR}prediction_test.tif', mode='wb') as localfile:
            localfile.write(api_result.content)
    
    #the data we want is in api_result['content']
    dem_read_in = gdal.Open(f'{OUTPUT_TIF_DIR}prediction_test.tif')

    hillshade_output = gdal.DEMProcessing(f'{OUTPUT_TIF_DIR}prediction_test_hillshade.tif', dem_read_in, 'hillshade',
                            altitude=45, azimuth=315)

    dem = gdal.Open(f'{OUTPUT_TIF_DIR}prediction_test_hillshade.tif')

    width = dem.RasterXSize
    height = dem.RasterYSize

    create_tiles(xsize=width,
                ysize=height,
                out_path=TILED_TIF_OUTPATH,
                in_path=OUTPUT_TIF_DIR,
                input_filename='prediction_test_hillshade.tif')
    