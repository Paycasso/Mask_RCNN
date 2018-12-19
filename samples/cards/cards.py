import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import pandas as pd
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from samples.cards.synthdocs import rounded_corners, random_homography, coordinate_geometry, photo_synthesizer


class CardsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cards"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 category for documents

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 42, 64, 84, 128) # try this instead

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 0.65, 0.8]

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (112, 112)  # (height, width) of the mini-mask


RANDOM_HOMOGRAPHY_CONIG = {'scale_amount' : (1, 2),
                           'warp_amount': 0.0005,
                           'rotation_amount' : 15,
                           'translate_amount' : 100,}

PHOTO_SYNTHESIZER_CONFIG = {'hsv_jitter_amount': (2, 10, 90),
                            'blur_param':0.25}


class CardsDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.via_cornerpoints_filepath = kwargs.get('via_cornerpoints_filepath')
        self.real_image_dirpath = kwargs.get('real_image_dirpath')
        self.template_dirpath = kwargs.get('template_dirpath')
        self.backgrounds_dirpath = kwargs.get('backgrounds_dirpath')
        self.prob_real = kwargs.get('prob_real')

        self.df_real_images = pd.read_csv(self.via_cornerpoints_filepath)
        self.card_template_filenames = utils.get_all_file_list(self.template_dirpath)
        self.background_image_filenames = utils.get_all_file_list(self.backgrounds_dirpath)

        # self.width, self.height = kwargs.get('width'), kwargs.get('height')

        self.homography_creator = random_homography.Random_Homography(**RANDOM_HOMOGRAPHY_CONIG)
        self.photographer = photo_synthesizer.PhotoSynthesizer(PHOTO_SYNTHESIZER_CONFIG['hsv_jitter_amount'],
                                                               PHOTO_SYNTHESIZER_CONFIG['blur_param'])

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("cards", 1, "card")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            random_image_properties = self.random_image(height, width)
            self.add_image("cards", image_id=i, path=None,
                           width=width, height=height,
                           cards=['card'], # to fit the previous "shapes" interface that requires a list of
                                           # shapes present in the image

                           # bg_color=bg_color, shapes=shapes

                           # cornerpoints=cornerpoints,
                           # real=real,
                           # real_image_path=real_image_path,
                           # background_image_path=background_image_path,
                           # card_template_path=card_template_path,

                           **random_image_properties
                           )

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        # bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        # image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        # image = image * bg_color.astype(np.uint8)
        # for shape, color, dims in info['shapes']:
        #     image = self.draw_shape(image, shape, dims, color)

        width, height = info['width'], info['height']

        if info['real']:
            # load image from disk
            impath = os.path.join(self.real_image_dirpath, info['real_image_path'])
            image = cv2.imread(impath,1)
            image = cv2.resize(image, (width, height), cv2.INTER_CUBIC)
        else:
            # synthesize image
            background_path = info['background_image_path']
            card_template_path = info['card_template_path']
            cornerpoints = info['cornerpoints']
            image = self.synthesize_image(card_template_path, background_path, cornerpoints, (width, height))
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cards":
            return info["cards"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for num_categories of the given image ID.
        """
        info = self.image_info[image_id]
        num_cards = info['cards']
        # count = len(num_cards)
        count = 1 # there will only ever be 1 card per image (for simplicity) TODO: do multiple documents?
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        # for i, _ in enumerate(info['cards']):
        mask[:, :, 0] = self.draw_quadrilateral(mask[:, :, 0].copy(), info['cornerpoints'], 1)

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        # class_ids = np.array([self.class_names.index(s[0]) for s in num_categories])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def draw_quadrilateral(self, image, cornerpoints, color):
        cornerpoints = np.int32([cornerpoints])
        mask = cv2.fillPoly(image, cornerpoints, color)
        return mask

    def random_image(self, height, width):
        """Creates random specifications of an image containing a document.
        Returns whether the image is real or synthetic, and a collection of
        specifications that can be used to draw/load the image.
        """
        random_image_properties = {}
        # flip a coin to determine whether image should be synthetic or real
        if random.random() < self.prob_real:
            random_image_properties['real'] = True
            # select a random row from the list of filenames
            # random_row = self.df_real_images.sample(n=1)
            # print(random_row)
            # filename, x0, y0, x1, y1, x2, y2, x3, y3 = random_row
            random_index = random.choice(list(range(len(self.df_real_images))))
            filename = self.df_real_images['filename'].values[random_index]
            print(filename)
            x0 = self.df_real_images['x0'].values[random_index]
            y0 = self.df_real_images['y0'].values[random_index]
            x1 = self.df_real_images['x1'].values[random_index]
            y1 = self.df_real_images['y1'].values[random_index]
            x2 = self.df_real_images['x2'].values[random_index]
            y2 = self.df_real_images['y2'].values[random_index]
            x3 = self.df_real_images['x3'].values[random_index]
            y3 = self.df_real_images['y3'].values[random_index]
            random_image_properties['real_image_path'] = filename
            cornerpoints = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]], dtype=np.float32)

            abs_impath = os.path.join(self.real_image_dirpath, filename)
            im = cv2.imread(abs_impath, 1)
            print('Loaded image from {}'.format(abs_impath))
            h,w,c = im.shape
            sfx = float(width)/w
            sfy = float(height)/h
            scale_matrix = np.array([[sfx,0,0],
                                     [0,sfy,0],
                                     [0,0,1]], dtype=np.float32)

            # mask = np.zeros((h,w))
            # mask = cv2.fillPoly(mask, np.int32([cornerpoints]), 1)
            # utils.showims([im, mask], ['im', 'mask'])

            # im_shrunk = cv2.resize(im, (width, height))
            # mask_shrunk=np.zeros((height, width))

            cornerpoints_shrunk = coordinate_geometry.transform_cornerpoints_2D(scale_matrix, cornerpoints)
            cornerpoints = cornerpoints_shrunk

            # mask_shrunk = cv2.fillPoly(mask_shrunk, np.int32([cornerpoints_shrunk]), 1)
            # utils.showims([im_shrunk, mask_shrunk], ['im_shrunk', 'mask_shrunk'])

            # cornerpoints = coordinate_geometry.transform_cornerpoints_2D(scale_matrix, cornerpoints)
            random_image_properties['cornerpoints'] = np.int32(cornerpoints)


            # and set unused variables to None
            random_image_properties['card_template_path'] = None
            random_image_properties['background_image_path'] = None
        else:
            random_image_properties['real'] = False
            random_image_properties['card_template_path'] = random.choice(self.card_template_filenames)
            random_image_properties['background_image_path'] = random.choice(self.background_image_filenames)
            random_image_properties['cornerpoints'] = self.random_cornerpoints(height, width)

            # and set unused variables to NoneX
            random_image_properties['real_image_path'] = None

        return random_image_properties

    def random_cornerpoints(self, height, width):
        h = self.homography_creator.random_homography((width, height))
        original_cornerpoints = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
        cornerpoints = coordinate_geometry.transform_cornerpoints_2D(h, original_cornerpoints)
        return cornerpoints

    def synthesize_image(self, card_template_path, background_image_path, cornerpoints, canvas_size):
        width, height = canvas_size
        original_cornerpoints = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
        cornerpoints = np.float32(cornerpoints)
        h = cv2.getPerspectiveTransform(cornerpoints, original_cornerpoints)
        im_card = cv2.imread(card_template_path)
        im_background = cv2.imread(background_image_path)
        photo, homography, cornerpoints_warped = self.photographer.snap_this_homography(im_card, im_background, canvas_size, h)
        return photo


if __name__ == '__main__':


    CONFIG = {
    'via_cornerpoints_filepath': 'classified_quadrilateral_cornerpoints.txt',
    'real_image_dirpath': os.path.join(os.path.expanduser('~'), 'Paycasso_Data', 'Classified_Images', 'C'),
    'template_dirpath': os.path.join(os.path.expanduser('~'), 'Paycasso_Data', '_1355_20161123'),
    'backgrounds_dirpath': os.path.join(os.path.expanduser('~'), 'Paycasso_Data', 'BACKGROUND_IMAGES'),
    'prob_real': 0.5,
    'width': 320,
    'height': 240
    }


    # Training dataset
    dataset_train = CardsDataset(**CONFIG)
    dataset_train.load_shapes(13, CONFIG['height'], CONFIG['width'])
    dataset_train.prepare()

    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)