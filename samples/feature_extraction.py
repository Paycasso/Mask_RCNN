import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.cards.cards import CardsConfig

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs","cards20181217T1222/mask_rcnn_cards_0002.h5") # meh
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", "cards20181217T1337/mask_rcnn_cards_0004.h5") # not bad
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", "cards20181217T1337/mask_rcnn_cards_0011.h5") # OK
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", "cards20181218T2246/mask_rcnn_cards_0004.h5") # meh
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", 'cards20181218T2346/mask_rcnn_cards_0006.h5') # not bad at all!
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", 'cards20181218T2346/mask_rcnn_cards_0010.h5') # prob best so far?
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", 'cards20181219T1303/mask_rcnn_cards_0018.h5') # pretty good!
WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", 'cards20181219T1303/mask_rcnn_cards_0050.h5') # pretty damn good


# # Download COCO trained weights from Releases if needed
# if not os.path.exists(WEIGHTS_PATH):
#     utils.download_trained_weights(WEIGHTS_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = '/home/sal9000/Paycasso_Data/LivePaycassoTestImages'
# IMAGE_DIR = '/home/sal9000/Paycasso_Data/NZ_test_images'
# IMAGE_DIR = '/home/sal9000/Paycasso_Data/Classified_Images/C2'
# IMAGE_DIR = '/home/sal9000/Pictures'

class InferenceConfig(CardsConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 1
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.BATCH_SIZE = 1
config.display()

# Create model object in inference mode.
# model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model = modellib.MaskRCNNFeatureExtraction(mode='feature_extraction', model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(WEIGHTS_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']
class_names = ['BG', 'Document']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
for file_name in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

    start_time = time.time()
    # Run detection
    results = model.detect([image], verbose=1)
    print('Detection took {} seconds'.format(time.time()-start_time))
    print(results.shape)
    # print(results)

    #
    # # Visualize results
    # r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])