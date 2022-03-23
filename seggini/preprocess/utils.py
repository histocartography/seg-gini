import os
import numpy as np
import h5py
from .constants import *
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000000

def create_data_pickle(
        dir_name: Path,
        columns: list(),
        save_path: Path,
        constants: Constants
):
    df = pd.DataFrame(columns=columns)
    for i, row in constants.IMAGE_LABELS.iterrows():
        name = row[0]
        path = dir_name / (name +'.png')

        if os.path.isfile(path):
            df_ = pd.DataFrame([[name, Path(path)]], columns=columns)
            df = pd.concat([df, df_], axis=0, ignore_index=True)

    df.set_index('name', inplace=True)
    df.to_pickle(save_path)

def create_annotation_pickle(
        dir_name: Path,
        columns: list(),
        save_path: Path,
        constants: Constants
):
    df = pd.DataFrame(columns=columns)

    for i, row in constants.IMAGE_LABELS.iterrows():
        name = row[0]
        path = dir_name / (name +'.png')

        if os.path.isfile(path):
            data_provider = row[2]
            gleason_grade = row[3].split('+')

            label = np.zeros(NR_CLASSES, dtype=int)
            label[LABEL_TO_INDEX[gleason_grade[0]]] = 1
            label[LABEL_TO_INDEX[gleason_grade[1]]] = 1
            annotation = [name, data_provider] + label.tolist()
            df_ = pd.DataFrame([annotation], columns=columns)
            df = pd.concat([df, df_], axis=0, ignore_index=True)

    df.set_index('name', inplace=True)
    df.to_pickle(save_path)

def create_pickle(constants: Constants):
    os.makedirs(constants.PICKLE_PATH, exist_ok=True)

    # Create images pickle
    create_data_pickle(
        dir_name=constants.IMAGES_PATH,
        columns=['name', 'image_path'],
        save_path=constants.PICKLE_PATH / Path('images.pickle'),
        constants=constants
    )
    
    # Create annotation masks pickle
    for partial in PARTIAL:
        create_data_pickle(
            dir_name=constants.ANNOTATIONS_PATH/ Path('annotation_masks_' + str(partial)),
            columns=['name', 'annotation_mask_path'],
            save_path=constants.PICKLE_PATH / Path('annotation_masks_' + str(partial) + '.pickle'),
            constants = constants
        )

    # Create labels pickle
    create_annotation_pickle(
        dir_name=constants.IMAGES_PATH,
        columns=['name', 'data_provider', 'benign', 'grade3', 'grade4', 'grade5'],
        save_path=constants.PICKLE_PATH / Path('image_level_annotations.pickle'),
        constants=constants
    )

def save_tissue_mask(
        mask: np.ndarray,
        path: Path
):
    color_palette = [255, 255, 255,     # non-tissue is white
                     0, 100, 0]         # tissue is green
    mask = Image.fromarray(mask.astype('uint8'), 'P')
    mask.putpalette(color_palette)
    mask.save(path)

def save_superpixel_map(
        maps: np.ndarray,
        path: Path
):
    output_key = "default_key"

    with h5py.File(path, "w") as f:
        if not isinstance(maps, tuple):
            maps = tuple([maps])

        for i, output in enumerate(maps):
            f.create_dataset(
                f"{output_key}_{i}",
                data=output,
                compression="gzip",
                compression_opts=9,
            )