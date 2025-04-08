'''
Get DEM data. Via opentopo: 300 calls/24 hours
1m data is 250km^2 for spatial limit

Approach: for all unique coords in training data grab the DEM
But do not grab repeats.. if that makes sense

Note that API has a 300/day limit and will error out oddly if met

'''

import pyproj
import requests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None
from pyproj import Transformer
import geopandas as gpd
import sys
import time

sys.path.append('C:/Users/jlomb/Documents/PersonalProjects/MPExtensions')

from scrapingData.openBetaScraper import parseAllFiles

from scrapingData.webScraperTools import removeIncorrectLatLng, add_url

from scrapingData.qaqcOpenBetaData import removeByKeyword, \
    duplicateGPSRemover, modifyNameForOutput

from generatingData.neighboringPoints import transformCoords, findTreePairs, \
    updateAreaWithPairs, extractPointsWithNeighbors

from scrapingData.read_manual_gps_boulders import manual_lat_long
from scrapingData.get_vedauwoo_spatial import grab_vedauwoo_data

from os import urandom

OPEN_TOPO_KEY = '28c874f8b6428e05ba6a349792c629aa'
#OPEN_TOPO_KEY_2 = 'f28cca682f98d6842ccca4b202b61529'
def ping_open_topo_api(bbox:pd.DataFrame,
                       dataset_type='USGS1m'):

    payload = {
        'API_Key':OPEN_TOPO_KEY,
        'south':bbox['south'],
        'west':bbox['west'],
        'east':bbox['east'],
        'north':bbox['north'],
        'outputFormat':'GTiff',
        'datasetName':dataset_type
    }

    BASE_API_URL = 'https://portal.opentopography.org/API/usgsdem'

    try:
        access_api = requests.get(BASE_API_URL,
                                  params=payload)
        
    except Exception as e:
        return e

    return access_api


def find_missing_neighbors(boulder_data:pd.DataFrame) -> pd.DataFrame:

    '''
    Iterate through a dataframe in order to find what neighbors appear to be
    'far away' for a given area. Note that 'far away' is defined elsewhere.
    '''

    major_area_names = boulder_data.major_area.unique()

    area_list = []
    for area in major_area_names:

        sub_area = boulder_data[boulder_data.major_area == area]
        complete_list = sub_area['index'].to_list()

        missed_neighbors = []
        for _, row in sub_area.iterrows():

            missed_neighbor = [x for x in complete_list if x not in row.neighbors]
            missed_neighbor = [x for x in missed_neighbor if x != row['index']]
            missed_neighbors.append(missed_neighbor)

        sub_area['missed_neighbors'] = missed_neighbors

        area_list.append(sub_area)

    areas_updated = pd.concat(area_list)

    return areas_updated


def generate_mcp_ids(data:pd.DataFrame) -> pd.DataFrame:

    '''
    For all major areas look through the unique set of missed neighbors.
    Generate a MCP id for each of these in order to generate data to 
    feed to the DEM API
    '''

    major_area_names = data.major_area.unique()
    area_list = []
    for area in major_area_names:

        sub_area = data[data.major_area == area]

        #Find each unique set of missed_neighbors
        unique_missing_neighbors = [list(x) for x in set(tuple(x) for x in sub_area.missed_neighbors)]
        sub_area_assign = {index: data for index, data in enumerate(unique_missing_neighbors)}

        stored_list = []
        for neighbors_list in sub_area.missed_neighbors:
            matching_key = [key for key, value in sub_area_assign.items() if value == neighbors_list]
            stored_list.append(matching_key)

        sub_area['mcp_id'] = [str(item) + f'_{area}' for items in stored_list for item in items]
    
        area_list.append(sub_area)

    all_areas = pd.concat(area_list)
    return all_areas


def get_bbox_values(mcp_data: pd.DataFrame,
                    buffer_amount=0.005) -> pd.DataFrame:

    '''
    For a set of MCPs get bbox vaslues
    Buffer amount is expressed in degrees. So 0.01 is about a km buffer on all sides
    
    '''

  # Now find the corners of the bboxs
    mcp_unique_vals = mcp_data.mcp_id.unique()
  
    mcp_data['lon'] = [x[1] for x in mcp_data['lnglat']]
    mcp_data['lat'] = [x[0] for x in mcp_data['lnglat']]

    bbox_values = pd.DataFrame()
    for index, mcp_id in enumerate(mcp_unique_vals):

        sub_area = mcp_data[mcp_data.mcp_id == mcp_id]

        gdf = gpd.GeoDataFrame(sub_area,
                               geometry=gpd.points_from_xy(x=sub_area.lat,
                                                           y=sub_area.lon))
    
        convex_hull_bounds = gdf.convex_hull.total_bounds
        # 0 -> min_x, 1 -> min_y, 2 -> max_x, 3 -> max_y
        #Hardcoded for lat long.
        #Due to inaccuracy of measurements add a buffer so we capture the bbox
        #plus more.
        convex_df = pd.DataFrame({
            'east':convex_hull_bounds[3] + buffer_amount, #max_x
            'south':convex_hull_bounds[0] - buffer_amount, #min_x
            'west':convex_hull_bounds[1] - buffer_amount, #min_y
            'north':convex_hull_bounds[2] + buffer_amount, #max_x
            'id':mcp_id
        }, index=[index])

        bbox_values = pd.concat([bbox_values,convex_df])

    return bbox_values


def export_training_data(boulders:list,
                         output_dir:str):

    to_output = pd.DataFrame(boulders)
    to_output['lon'] = [x[1] for x in to_output['lnglat']]
    to_output['lat'] = [x[0] for x in to_output['lnglat']]

    trans = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:26913",
        always_xy=True,
    )
    xx, yy = trans.transform(to_output["lon"].values, to_output["lat"].values)

    to_output.drop(columns=['lnglat'], inplace=True)
    to_output['x'] = xx
    to_output['y'] = yy

    gdf = gpd.GeoDataFrame(to_output,
                                geometry=gpd.points_from_xy(x=to_output.x,
                                                            y=to_output.y))
    
    gdf=gdf.set_crs(26913)
    
    gdf.to_file(f'{output_dir}hq_training_data_utm.shp', driver='ESRI Shapefile')


def manually_remove(boulders:list) -> list:

    '''
    From manual inspection JAL saw some obvious outliers.
    '''

    unique_ids_to_remove = [110146951, 113476988, 106433739, 106492511,
                            106111459, 253, 254, 255, 256, 257, 258]
    
    unique_ids = [str(x) for x in unique_ids_to_remove]
    
    clean_boulders = [x for x in boulders if str(x['unique-id']) not in unique_ids]

    return clean_boulders


def generate_mcp_ids(data:pd.DataFrame) -> pd.DataFrame:

    '''
    For all major areas look through the unique set of missed neighbors.
    Generate a MCP id for each of these in order to generate data to 
    feed to the DEM API
    '''

    major_area_names = data.major_area.unique()
    area_list = []
    for area in major_area_names:

        sub_area = data[data.major_area == area]

        #Find each unique set of missed_neighbors
        unique_missing_neighbors = [list(x) for x in set(tuple(x) for x in sub_area.missed_neighbors)]
        sub_area_assign = {index: data for index, data in enumerate(unique_missing_neighbors)}

        stored_list = []
        for neighbors_list in sub_area.missed_neighbors:
            matching_key = [key for key, value in sub_area_assign.items() if value == neighbors_list]
            stored_list.append(matching_key)

        sub_area['mcp_id'] = [str(item) + f'_{area}' for items in stored_list for item in items]
    
        area_list.append(sub_area)

    all_areas = pd.concat(area_list)
    return all_areas


def generate_major_area_id(boulder_neighbors: pd.DataFrame) -> pd.DataFrame:

    boulder_neighbors['major_area'] = ''

    all_neighbors = boulder_neighbors['neighbors']

    all_neighbors = all_neighbors.to_list()
    all_neighbors = [set(x) for x in all_neighbors]

    for _, data in boulder_neighbors.iterrows():

        current_neighbors = set(data['neighbors'])

        if data['major_area'] == '':
            index_matches = [True if len(current_neighbors.intersection(x)) > 0 else False for x in all_neighbors]

            boulder_neighbors.loc[index_matches ,'major_area'] = urandom(5).hex()

    return boulder_neighbors


if __name__ == '__main__':

    SHAPEFILE_DIR = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/training_data/'
    OUTPUT_DIR = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/dem/'
    BASE_DIR = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/web_json'

    full_data = parseAllFiles(fileParent=BASE_DIR + '/hand_picked/')

    unlist_data = [item for sublist in full_data for item in sublist]

    #add in the manual data (the ones JAL hand entered)
    HAND_DIR = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/gps_hand_entered/'

    hand_entered = manual_lat_long(HAND_DIR)

    vedauwoo_data = grab_vedauwoo_data()

    combined_data = unlist_data + hand_entered + vedauwoo_data

    area_clean = removeIncorrectLatLng(combined_data)

    remove_dupl = duplicateGPSRemover(listOfAreas=area_clean)

    remove_dupl = removeByKeyword(listOfAreas=remove_dupl,
        keywords = ['Concrete','Building','Builder','CentralPark','Downtown'])

    correct_area_gps = modifyNameForOutput(remove_dupl)

    cleaned_boulders = manually_remove(correct_area_gps)

    #Unsure if add_url and transform coords matter
    add_url(cleaned_boulders)

    #Note that some of the DEMs will have no data.
    export_training_data(cleaned_boulders,
                         output_dir=SHAPEFILE_DIR)

    #Also export the lat long version
    to_output = pd.DataFrame(cleaned_boulders)
    to_output['lon'] = [x[1] for x in to_output['lnglat']]
    to_output['lat'] = [x[0] for x in to_output['lnglat']]

    to_output.drop(columns=['lnglat'], inplace=True)
    
    gdf = gpd.GeoDataFrame(to_output,
                                geometry=gpd.points_from_xy(x=to_output.lon,
                                                            y=to_output.lat))
    
    gdf=gdf.set_crs(4236)
    
    gdf.to_file('C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/training_data/training_data_hq.shp', driver='ESRI Shapefile')

    kd_coords = transformCoords(listOfPlaces=cleaned_boulders)

    #note that the area requirment for DEM is 25km2
    closest_pairs = findTreePairs(kd_coords,
                                  distanceValue=10)

    paired_areas = updateAreaWithPairs(listOfPlaces=correct_area_gps, closestPairs = closest_pairs)

    areas_with_neighbors = extractPointsWithNeighbors(paired_areas, neighborCount=1)

    #Convert to a dataframe to do a per major_area operation
    areas_with_neighbors = pd.DataFrame(areas_with_neighbors)

    #For now since I jsut want rmnp
    #areas_with_neighbors = areas_with_neighbors[areas_with_neighbors['major_area'] == 'rmnp']
    # areas_with_neighbors = generate_major_area_id(areas_with_neighbors)

    areas_updated = find_missing_neighbors(areas_with_neighbors)

    # #There are some that are far away. These are bishop + poudre. These are fine after manual review.
    # missed_info = [True if len(x) > 0 else False for x in areas_updated.missed_neighbors]
    # to_inv = areas_updated[missed_info]

    mcp_values = generate_mcp_ids(areas_updated)

    bbox_values = get_bbox_values(mcp_data=mcp_values)

    bbox_values['no_data'] = 0

    # #Output something so I can do x amount and then more every 24 hours
    #bbox_values.to_pickle('complete_bbox_values.pkl')

    #TODO: over api limit need to rerun
    #update this based on last thing attempted
    # bbox_values = pd.read_pickle('complete_bbox_values.pkl')
    # to_start = 1715
    
    # for index, row in bbox_values.iterrows():
    i = 0
    for i in range(i,len(bbox_values)):

        print(f'about to scrape: {bbox_values.iloc[i]}')


        row = bbox_values.iloc[i]
        #Reached limit
        if i > 1896:
            break

        api_result = ping_open_topo_api(bbox=row,
                                  dataset_type='USGS1m')
        
        
        if api_result.status_code != 200:
            time.sleep(1)

            api_result = ping_open_topo_api(bbox=row,
                                            dataset_type='USGS1m')

            if api_result.status_code != 200:
                bbox_values['no_data'][i] = 1
                print(f'The following row failed {row}')
                
                continue
            
        with open(f'{OUTPUT_DIR}{row.id}_hq_dem.tif', mode='wb') as localfile:
            localfile.write(api_result.content)

