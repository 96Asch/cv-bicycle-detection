import os
import sys
import argparse
import numpy as np
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class DelftBikesConfig(Config):
    NAME = "delft-bikes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # background + bike

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 64


class DelftBikesDataSet(utils.Dataset):
    pass


def setup_parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', required=True, type=str, help='Path to trained weights')
    # parser.add_argument('--input', required=True, type=str, help='Path to dataset')
    # parser.add_argument('--exps', required=True, type=str, help='Path to store trained model/logs')
    return parser.parse_args()


def main(argv):
    config = DelftBikesConfig()



if __name__ == '__main__':
    argv = setup_parse()
    main(argv)
