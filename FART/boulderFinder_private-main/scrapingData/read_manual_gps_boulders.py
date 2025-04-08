'''
JAL got gps coordinates form guidegooks and put into excel.
read these in and convert to a 'friendly' format

JAL didn't type out the stuff before the decimal place. that's what
some of this translation does
'''

import pandas as pd

##TODO: bug with modify manual area not doing anything
def modify_manual_area(area_data:pd.DataFrame) -> pd.DataFrame:

    modify_key = {
        'telluride_108':'telluride',
        'telluride_38_108':'telluride',
        'telluride':'telluride',
        'red feather':'red_feather',
        'rmnp':'rmnp'
    }

    area_names = area_data['major_area'].to_list()
    area_data['major_area'] = [modify_key.get(x, '') for x in area_names]
    return area_data

def manual_lat_long(data_dir:str) -> list[dict]:
    
    hand_entered = pd.read_csv(data_dir+'spatial_data_bouldering_update.csv')

    translation = pd.DataFrame({
        'lat':[40,37,37,38,40],
        'long':[-105,-107,-108,-108,-105],
        'major_area':['red feather','telluride','telluride_108','telluride_38_108',
                      'rmnp']
    })

    lat_longs = []
    for _, row in hand_entered.iterrows():
        
        area_key = row['major_area']
        to_look_up = translation[translation.major_area == area_key]

        lat_lon = [str(to_look_up.lat.iloc[0]) + '.' + str(row.lat), 
                str(to_look_up.long.iloc[0]) + '.' + str(row.long),]
        
        lat_lon = [float(x) for x in lat_lon]

        lat_longs.append(lat_lon)

    hand_entered['lnglat'] = lat_longs

    modified_hand_entered = modify_manual_area(hand_entered)

    output = modified_hand_entered[['boulder_name','major_area','lnglat']]

    output['parent-id'] = output['major_area']
    output['area_name'] = output['boulder_name']
    output['us_state'] = 'colorado'
    output['unique-id'] = output.index.values
    output.drop(columns=['boulder_name'], inplace=True)

    list_format = output.to_dict(orient='records')

    return list_format
