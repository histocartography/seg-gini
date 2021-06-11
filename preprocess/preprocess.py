import glob
from tqdm import tqdm
from dgl.data.utils import save_graphs
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000000

from utils import *
from histocartography.preprocessing import (
    MacenkoStainNormalizer,              # stain normalizer
    GaussianTissueMask,                  # tissue mask detector
    ColorMergedSuperpixelExtractor,      # superpixel extractor
    AugmentedDeepFeatureExtractor,       # feature extractor
    AnnotationPostProcessor,             # processing annotations
    RAGGraphBuilder                      # graph builder,
)

def preprocessing():
    # get image path
    image_fnames = glob.glob(os.path.join(
        IMAGES_PATH,
        '*.png'))
    image_fnames.sort()

    # 1. define stain normalizer
    normalizer = MacenkoStainNormalizer(target_path=STAIN_NORM_TARGET_IMAGE)

    # 2. define tissue detection
    tissue_detector = GaussianTissueMask(downsampling_factor=2)

    # 3. define superpixel extractor
    spx_extractor = ColorMergedSuperpixelExtractor(
        superpixel_size=1000,
        max_nr_superpixels=15000,
        blur_kernel_size=3,
        compactness=20,
        threshold=0.01,
        downsampling_factor=4
    )

    # 4. define feature extractor
    feature_extractor = AugmentedDeepFeatureExtractor(
        architecture='mobilenet_v2',
        num_workers=1,
        rotations=[0, 90],
        flips=['n', 'h'],
        patch_size=144,
        stride=144,
        resize_size=224,
        batch_size=512,
    )

    # 5. define annotation post-processor
    annotation_post_processor = AnnotationPostProcessor(background_index=4)

    # 6. define graph builder
    graph_builder = RAGGraphBuilder(
        nr_annotation_classes=5,
        annotation_background_class=4,
        add_loc_feats=False
    )

    # 7. define var to store image IDs that failed (for whatever reason)
    image_ids_failing = []

    # 8. process all the images
    for image_path in tqdm(image_fnames):

        # a. load image & check if already there
        image_name = os.path.basename(image_path).split('.')[0]
        image = np.array(Image.open(image_path))
        print('Processing...', image_name)

        # b. stain norm the image
        try:
            normalized_image = normalizer.process(image)
        except:
            print('Warning: {} failed during stain normalization.'.format(image_path))
            image_ids_failing.append(image_path)
            pass

        # c. extract tissue mask
        try:
            tissue_mask = tissue_detector.process(normalized_image)
            save_tissue_mask(tissue_mask, TISSUE_MASKS_PATH / (image_name + '.png'))
        except:
            print('Warning: {} failed during tissue mask detection.'.format(image_path))
            image_ids_failing.append(image_path)
            pass

        # d. extract superpixels
        try:
            superpixels, _ = spx_extractor.process(normalized_image, tissue_mask)
            save_superpixel_map(superpixels, SUPERPIXELS_PATH / (image_name + '.h5'))
        except:
            print('Warning: {} failed during superpixel extraction.'.format(image_path))
            image_ids_failing.append(image_path)
            pass

        # e. extract features
        try:
            features = feature_extractor.process(normalized_image, superpixels)
        except:
            print('Warning: {} failed during feature extraction.'.format(image_path))
            image_ids_failing.append(image_path)
            pass

        print('HERE')
        exit()

        # f. load annotation masks
        for partial in PARTIAL:
            annotation_path = ANNOTATIONS_PATH / \
                              ('annotation_masks_' + str(partial)) / \
                              (image_name + '.png')
            annotation_mask = np.array(Image.open(annotation_path))

            # g. annotation post-processing
            try:
                annotation_mask = annotation_post_processor.process(annotation_mask, tissue_mask)
            except:
                print('Warning: {} failed during annotation post-processing.'.format(image_path))
                image_ids_failing.append(image_path)
                pass

            # h. graph building
            try:
                graph = graph_builder.process(superpixels, features, annotation_mask)

            except:
                print('Warning: {} failed during graph building.'.format(image_path))
                image_ids_failing.append(image_path)
                pass

            # i. save
            save_graphs(
                filename=str(GRAPHS_PATH/ ('partial_' + str(partial)) / (image_name + '.bin')),
                g_list=[graph]
            )

if __name__ == "__main__":
    create_pickle()
    preprocessing()