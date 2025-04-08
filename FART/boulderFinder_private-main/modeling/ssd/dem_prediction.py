'''
This script takes a set of coordinates and outputs images that show predictions

'''

from torch_snippets import *
from os import listdir
from os.path import isfile, join
from PIL import Image
from modeling.ssd.model_utils import SSD300
import torch
from non_dem.sliceImages import detect_a_slice, untile_bbs

MODEL_LOC = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/models/'
TILED_TIF_OUTPATH = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/rawData/dem_predict/tiled/'

def unlist_data(listed_data):
    unlist = [item for sublist in listed_data for item in sublist]
    return unlist

##DEBUG HERE. note that i set min_score to be quite low in detect_a_slice

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SSD300(2,device)
    checkpoint = torch.load(MODEL_LOC + 'SSD300_DEM_test_two_200_ep.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    tiled_files = [x for x in listdir(TILED_TIF_OUTPATH) if isfile(join(TILED_TIF_OUTPATH, x))]

    bbs_list = []
    labels_list = []
    scores_list = []
    for tile in tiled_files:
          
        img = Image.open(TILED_TIF_OUTPATH + tile).convert("RGB")

        bbs, labels, scores = detect_a_slice(sliced_image=img,
                                             model=model, device=device)

        if 'Boulder' in labels:
            bbs_list.append(bbs)
            labels_list.append(labels)
            scores_list.append(scores)

    unlisted_boxes = unlist_data(bbs)
    unlisted_labels = unlist_data(labels)
    unlisted_scores = scores

    label_with_conf = [f'{l} @ {s:.2f}' for l,s in zip(unlisted_labels,unlisted_scores)]

    show(img, bbs=bbs)