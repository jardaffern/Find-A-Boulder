from PIL import Image
from torch_snippets.loader import show
import pandas as pd

FINAL_DEST = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/modeling/yolo/train_complete/'

#get the bboxes
# bboxs = pd.read_pickle('C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/complete_bbox_values.pkl')
# df = pd.read_pickle('C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/labeled_data/boulders_dem_tiled_yolo_complete.pkl')
# df.rename(columns={'x_min':'XMin',
#                     'x_max':'XMax',
#                     'y_min':'YMin',
#                     'y_max':'YMax'}, inplace=True)

one_image = Image.open(FINAL_DEST + 'images/0_0d7aa5a4d6_complete_dem_hillshade_tile_0_1280.png')
one_label = pd.read_csv(FINAL_DEST + 'labels/0_0d7aa5a4d6_complete_dem_hillshade_tile_0_1280.txt',
                        sep=' ', names = ['X','Y','width','height'])

one_label['XMin'] = ((one_label['X'] - one_label['width']/2) * 640).astype(int)
one_label['YMin'] = ((one_label['Y'] - one_label['height']/2) * 640).astype(int)
one_label['XMax'] = ((one_label['X'] + one_label['width']/2) * 640).astype(int)
one_label['YMax'] = ((one_label['Y'] + one_label['height']/2) * 640).astype(int)
one_label.index = [0,1,2,3,4,5,6]

to_store = []
for index, data in one_label.iterrows():
    print(index)
    one_boulder = one_label.values[index][4:8].tolist()
    one_boulder = [int(x) for x in one_boulder]
    to_store.append(one_boulder)

#output style is 
#XMin,YMin,XMax,YMax


#But JAL thinks it shoudl be (Xmid, Ymid, width, height)

show(one_image, bbs=to_store)

#TODO:
#view labels overlaid with images

#likely need to remove some of the training data if all the DEM is black?