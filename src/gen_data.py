import json
import os
import sys
import argparse
import random
from tabnanny import verbose
import numpy as np
import cv2
from PIL import Image
import datetime
import pycococreatortools as pct
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm import tqdm

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
print(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))

import mrcnn.model as modellib


sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN', 'samples', 'coco'))
import samples.coco.coco as coco



# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(coco.CocoConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

INFO = {
    "description": "Bicycle Dataset",
    "url": "https://github.com/96Asch/cv-bicycle-detection",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "96Asch",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'bicycle',
        'supercategory': 'vehicle',
    },
 
]

MULTIPLE_BIKE_P = 0.6
MODEL_DIR = os.path.join("..", "output")
#######################################
# Imgaug image transformations
ia.seed(2)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.2, 0.5), "y": (0.2, 0.5)},
        translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
    )

], random_order=True)

#######################################

def setupParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help='Path to trained weights')   
    parser.add_argument('--coco', required=True, type=str, help='Path to COCO dataset')   
    parser.add_argument('--input', required=True, type=str, help="Path to input images")
    parser.add_argument('--output', required=True, type=str, help='Path to store the generated images')   
    parser.add_argument('--ann', required=True, type=str, help='Path to the generated annotation file')   
    return parser.parse_args()




def main(argv) -> None:
    config = InferenceConfig()
    config.display()

    dataset = coco.CocoDataset()
    dataset.load_coco(dataset_dir=argv.coco, subset='val', year='2017', return_coco=True, auto_download=True)
    dataset.prepare()

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    weights_path = argv.model

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    image_id = 1
    segmentation_id = 1

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    for root, _, files in os.walk(argv.input):
        image_files = pct.filter_for_jpeg(root, files)
        print(len(image_files))

        def prepare_bike_image(path):
            # print(path)
            bicycle_img = cv2.imread(path)
            if bicycle_img is None:
                print(f"Empty image {path}")
                return None
            bicycle_img = cv2.resize(bicycle_img, (1024, 1024))
            bicycle_img = seq.augment_image(bicycle_img)
            return bicycle_img


        for image_file in tqdm(image_files):

            # Retrieve between 1-3 bike images from the dataset including the current one
            bicycles = []

            bicycle = prepare_bike_image(image_file)

            if bicycle is not None:
                bicycles.append(bicycle)
            else:
                continue

            
            if random.uniform(0, 1) > MULTIPLE_BIKE_P:
                for _ in range(random.randint(0,2)):
                    bicycle = prepare_bike_image(random.choice(image_files))
                    if bicycle is not None:
                        bicycles.append(bicycle)
            
            image_info = pct.create_image_info(
                image_id, os.path.basename(image_file), bicycles[0].shape)
            coco_output["images"].append(image_info)
            
            results = [model.detect([b_img], verbose=0)[0] for b_img in bicycles]

            # Retrieve a random COCO background image to use
            coco_image_id = random.choice(dataset.image_ids)
            coco_image, _, _, _, _ = modellib.load_image_gt(dataset, config, coco_image_id, use_mini_mask=False)
            coco_image = cv2.cvtColor(coco_image, cv2.COLOR_RGB2BGR)


            category_info = {'id': 1, 'is_crowd': False }
            binary_masks = []

            for result in results:
                binary_mask = result['masks'].astype(np.uint8).squeeze()
                num_detections = result['masks'].shape[2]

                if num_detections == 0:
                    continue

                # If there are more objects detected, only get the largest mask found
                if num_detections > 1:
                    max_mask_idx = np.argmax([np.sum(result['masks'][:, :, i]) for i in range(num_detections)])
                    binary_mask = result['masks'][:, :, max_mask_idx].astype(np.uint8)
                    if len(binary_mask.shape) == 3:
                        binary_mask = binary_mask.squeeze(axis=2)
                

                binary_masks.append(binary_mask.astype(np.uint8))

                annotation_info = pct.create_annotation_info(
                                segmentation_id, image_id, category_info, binary_mask,
                                tolerance=2)

                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                segmentation_id = segmentation_id + 1
            
            masked_image = coco_image
            for idx, mask in enumerate(binary_masks):
                masked_image *= (mask==0)[..., None]
                masked_image += cv2.bitwise_and(bicycles[idx], bicycles[idx], mask=mask)

            cv2.imwrite(f"{os.path.join(argv.output, os.path.basename(image_file))}", masked_image)

            image_id = image_id + 1
    
    with open(f'{argv.ann}', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
        

if __name__ == '__main__':
    parser = setupParse()
    main(parser)