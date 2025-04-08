'''
Convert vedauwoo spatial data that JAL made into a 'training' data format 
Really jsut need lat longs
'''

import geopandas as gpd

def grab_vedauwoo_data():
    DATA_DIR = "C:/Users/jlomb/Documents/Vedauwoo Guidebook/spatial_data/Boulder delineations.shp"

    vedauwoo_data = gpd.read_file(DATA_DIR)

    y_values = [round(x.centroid.x,7) for x in vedauwoo_data.geometry] #These are the longitudes -105ish
    x_values = [round(x.centroid.y,7) for x in vedauwoo_data.geometry] #These are the latitudes, positive

    vedauwoo_data['lnglat'] = [[x,y] for x, y in zip(x_values, y_values)]
    vedauwoo_data['us_state'] = 'wyoming'
    vedauwoo_data['parent-id'] = 'vedauwoo'
    vedauwoo_data['area_name'] = vedauwoo_data['Sub-Area']
    vedauwoo_data['unique-id'] = vedauwoo_data.index.values + 1000
    vedauwoo_data['major_area'] = 'vedauwoo'

    to_output = vedauwoo_data[['major_area','lnglat','us_state',
                               'parent-id','area_name','unique-id']]
    
    return to_output.to_dict(orient='records')