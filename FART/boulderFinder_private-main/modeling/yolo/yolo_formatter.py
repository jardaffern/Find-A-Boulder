#see https://docs.ultralytics.com/datasets/detect/
'''
This convert .tif files to rgb, as well as moves the label dataframe to]
.txt files for each img
'''
import pandas as pd
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np

if __name__ == '__main__':
    
    DATA_ROOT = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/labeled_data/'
    TILED_TIF = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/tiled/yolo/'

    df = pd.read_pickle(f'{DATA_ROOT}boulders_dem_tiled_yolo.pkl')
    df.rename(columns={'x_min':'XMin',
                       'x_max':'XMax',
                       'y_min':'YMin',
                       'y_max':'YMax'}, inplace=True)

    #Two 'concerning' things.
    #One is how did we get negative values? I think it's edge cases
    #we also get all 0's sometimes
    df[df['XMin'] < 0] = 0
    df[df['YMin'] < 0] = 0

    df = df[~(df['label'] == 0)]

    to_modify = df['XMax'] > 640
    to_modify_y = df['YMax'] > 640

    df.loc[to_modify, 'XMax'] = 640
    df.loc[to_modify_y, 'YMax'] = 640

    hillshade_list = [x for x in listdir(TILED_TIF) if isfile(join(TILED_TIF, x))]

    hillshade_list = [x for x in hillshade_list if not ('.aux.xml' in x)]

    for tif in hillshade_list:
        img_path = TILED_TIF + tif
        img = Image.open(img_path).convert("RGB")

        modify_tif_name = tif.replace('.tif','')

        img.save(TILED_TIF + 'rgb/' + modify_tif_name + '.png')

    
    #Now output the relevant parts of 'df' to a txt file
    source = df['source'].unique()

    # YOLO FORMAT: X midpoint, Y midpoint, width, height
    #Add some randomness even though i set some strict sizes here
    height_randomizer = np.random.choice([2,3,4,5,6], size=len(df), replace=True)
    width_randomizer = np.random.choice([2,3,4,5,6], size=len(df), replace=True)

    df['height'] = ((df['YMax'] - df['YMin']) + height_randomizer)/640
    df['width'] = ((df['XMax'] - df['XMin']) + width_randomizer)/640

    df['x_mid'] = ((df['XMin'] + df['XMax']) / 2)/640
    
    df['y_mid'] = ((df['YMin'] + df['YMax']) / 2)/640

    for file in source:
        to_output = df[df['source'] == file]

        to_output['yolo_label'] = 0

        to_output = to_output[['yolo_label','x_mid',
                               'y_mid','width','height']]
        
        file_clean = file.replace('.tif','')


        file_name = TILED_TIF + 'labels/' + file_clean + '.txt'
        
        with open(file_name, 'a') as f:
            f.write(to_output.to_string(header=False, index=False))
