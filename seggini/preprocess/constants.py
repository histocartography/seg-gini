from pathlib import Path
import pandas as pd

NR_CLASSES = 4
LABEL_TO_INDEX = {'0': 0, '3': 1, '4': 2, '5': 3}
PARTIAL = [100]

class Constants:
    def __init__(self, base_path: Path):
        self.BASE_PATH = base_path
        self.set_constants()

    def set_constants(self):
        self.PREPROCESS_PATH = self.BASE_PATH / 'preprocess'
        self.IMAGES_PATH = self.BASE_PATH / 'images'
        self.ANNOTATIONS_PATH = self.BASE_PATH / 'annotation_masks'

        self.STAIN_NORM_TARGET_IMAGE = '../data/target.png'  # define stain normalization target image.
        self.IMAGE_LABELS = pd.read_csv(self.BASE_PATH / 'image_labels.csv')

        self.TISSUE_MASKS_PATH = self.PREPROCESS_PATH / 'tissue_masks'
        self.SUPERPIXELS_PATH = self.PREPROCESS_PATH / 'superpixels'
        self.GRAPHS_PATH = self.PREPROCESS_PATH / 'graphs'

        if not self.PREPROCESS_PATH.exists():
            self.PREPROCESS_PATH.mkdir()
        if not self.TISSUE_MASKS_PATH.exists():
            self.TISSUE_MASKS_PATH.mkdir()
        if not self.SUPERPIXELS_PATH.exists():
            self.SUPERPIXELS_PATH.mkdir()
        if not self.GRAPHS_PATH.exists():
            self.GRAPHS_PATH.mkdir()
        for partial in PARTIAL:
            if not (self.GRAPHS_PATH / ('partial_' + str(partial))).exists():
                (self.GRAPHS_PATH / ('partial_' + str(partial))).mkdir()