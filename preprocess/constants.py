'''This module handles all general constants of the experiment'''
from pathlib import Path
import pandas as pd

NR_CLASSES = 4
LABEL_TO_INDEX = {'0': 0, '3': 1, '4': 2, '5': 3}
PARTIAL = [25, 100]

# BASE_PATH = Path('/Users/pus/Desktop/Projects/Data/Gleason/Release/SICAPv2')
BASE_PATH = Path('/dataP/pus/wss/data/Release/SICAPv2')

PREPROCESS_PATH = BASE_PATH / 'preprocess'
IMAGES_PATH = BASE_PATH / 'images'
ANNOTATIONS_PATH = BASE_PATH / 'annotation_masks'

STAIN_NORM_TARGET_IMAGE = '../data/target.png'  # define stain normalization target image.
IMAGE_LABELS = pd.read_csv(BASE_PATH / 'image_labels.csv')

TISSUE_MASKS_PATH = PREPROCESS_PATH / 'tissue_masks'
SUPERPIXELS_PATH = PREPROCESS_PATH / 'superpixels'
GRAPHS_PATH = PREPROCESS_PATH / 'graphs'

if not PREPROCESS_PATH.exists():
    PREPROCESS_PATH.mkdir()
if not TISSUE_MASKS_PATH.exists():
    TISSUE_MASKS_PATH.mkdir()
if not SUPERPIXELS_PATH.exists():
    SUPERPIXELS_PATH.mkdir()
if not GRAPHS_PATH.exists():
    GRAPHS_PATH.mkdir()
for partial in PARTIAL:
    if not (GRAPHS_PATH / ('partial_' + str(partial))).exists():
        (GRAPHS_PATH / ('partial_' + str(partial))).mkdir()