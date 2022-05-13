import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
print(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN', 'samples', 'coco'))
import coco

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(coco.CocoConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1




def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



# Compute VOC-style Average Precision
def compute_batch_ap(image_ids, model, config, dataset):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    return APs



def setupParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help='Path to trained weights')   
    parser.add_argument('--input', required=True, type=str, help='Path to dataset')   
    parser.add_argument('--exps', required=True, type=str, help='Path to store trained model/logs')   
    return parser.parse_args()


def main(argv):
    config = InferenceConfig()
    config.display()

    dataset = coco.CocoDataset()
    dataset.load_coco(dataset_dir=argv.input, subset='val', year='2017', return_coco=True, auto_download=False)

    dataset.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=argv.exps,
                              config=config)

    weights_path = argv.model

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)


    image_id = random.choice(dataset.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                        dataset.image_reference(image_id)))
    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    plt.savefig(f'{info["source"]}_display.png')

    # Draw precision-recall curve
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            r['rois'], r['class_ids'], r['scores'], r['masks'])
    visualize.plot_precision_recall(AP, precisions, recalls)

    plt.savefig(f'{info["source"]}_pr.png')


    # Pick a set of random images
    image_ids = np.random.choice(dataset.image_ids, 10)
    APs = compute_batch_ap(image_ids, model, config, dataset)
    print("mAP @ IoU=50: ", np.mean(APs))


if __name__ == '__main__':
    argv = setupParse()
    main(argv)
