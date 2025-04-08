'''
Select an arbitrary amount of data to move to the train/label locations
'''

from os import listdir
from os.path import isfile, join
import random
import shutil
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def black_pixel_perc(img: Image):

    pixel_form = list(img.getdata())

    black_counter = [1 if x == (0,0,0) else 0 for x in pixel_form]

    black_perc = int((sum(black_counter)/len(black_counter)*100))

    return black_perc

if __name__ == '__main__':

    FINAL_DEST = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/modeling/yolo/'
    RGB_YOLO = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/tiled/yolo/rgb/'
    TXT_YOLO = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/spatial_data/hillshade/tiled/yolo/labels/'

    img_names = [x for x in listdir(RGB_YOLO) if isfile(join(RGB_YOLO, x))]
    txt_names = [x for x in listdir(TXT_YOLO) if isfile(join(TXT_YOLO, x))]

    #First, remove images that have a large black pixel percentage
    pixel_count = []
    for img in img_names:
        dem_image = Image.open(RGB_YOLO + img)
        dem_bp_count = black_pixel_perc(dem_image)
        pixel_count.append(dem_bp_count)

    large_perc_threshold = [True if x>10 else False for x in pixel_count]
    large_perc_indices = [i for i, x in enumerate(large_perc_threshold) if x]

    img_names = [i for j, i in enumerate(img_names) if j not in large_perc_indices]

    #Only use labels that have valid iamges (i.e. no alrge black pixel perc)
    txt_names = [x for x in txt_names if x.replace('.txt','.png') in img_names]

    #ten percent goes to validation data
    sample_size = 10
    valid_data = random.sample(range(len(txt_names)), int(len(txt_names)/sample_size))

    #Extract just the validation labels
    valid_labels = []
    for i in valid_data:

        valid_labels.append(txt_names[i])

    train_labels = [x for x in txt_names if x not in valid_labels] 

    #Get the relevant images
    train_images = [x.replace('.txt','.png') for x in train_labels]
    valid_images = [x.replace('.txt','.png') for x in valid_labels]

    #Now most all the data
    for label, img in zip(train_labels, train_images):

        label_source = TXT_YOLO + label
        label_dest = FINAL_DEST + 'train/labels/' + label

        img_source = RGB_YOLO + img
        img_dest = FINAL_DEST + 'train/images/' + img

        #Move images and labels
        shutil.copy(label_source, label_dest)
        shutil.copy(img_source, img_dest)

    #Now most all the data
    for label, img in zip(valid_labels, valid_images):

        label_source = TXT_YOLO + label
        label_dest = FINAL_DEST + 'valid/labels/' + label

        img_source = RGB_YOLO + img
        img_dest = FINAL_DEST + 'valid/images/' + img

        #Move images and labels
        shutil.copy(label_source, label_dest)
        shutil.copy(img_source, img_dest)

    #NOTE: Add in some code to move some blank images to the train folder
    images_with_boulders = train_images + valid_images
    no_boulders = [x for x in img_names if x not in images_with_boulders]

    #Hardcode for now
    to_subset = random.sample(range(len(no_boulders)), int(len(no_boulders)/10))

    no_boulders = [no_boulders[i] for i in to_subset]

    #move just the blank images
    for img in no_boulders:

        img_source = RGB_YOLO + img
        img_dest = FINAL_DEST + 'train/images/' + img

        shutil.copy(img_source, img_dest)