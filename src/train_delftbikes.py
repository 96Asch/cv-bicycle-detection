import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from samples.coco.coco import CocoDataset, DEFAULT_DATASET_YEAR


class DelftBikesConfig(Config):
    NAME = "delft-bikes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # background + bike

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 64


class DelftBikesDataSet(CocoDataset):
    def load_delft_bikes(self, dataset_dir: str, subset: str):
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset,
                                                                DEFAULT_DATASET_YEAR))
        class_ids = coco.getCatIds(catNms=['BG', 'bicycle'])
        self.load_coco(dataset_dir, subset, class_ids=class_ids)


def setup_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help='Path to trained weights')
    parser.add_argument('--input', required=True, type=str, help='Path to dataset')
    parser.add_argument('--exps', required=True, type=str, help='Path to store trained model/logs')
    parser.add_argument('--mode', required=True, type=str,
                        help='Either "training" or "inference" mode')
    parser.add_argument('--device', required=False, type=str, default='/gpu:0',
                        help='The device to run on: either /cpu:0 or /gpu:0')
    parser.add_argument('--layers', required=True, type=str, default='heads',
                        help='Can be "heads" for just the final layer or "all" for all layers or '
                             'a regex matching the MRCNN layer names')
    return parser.parse_args()


def main(args):
    config = DelftBikesConfig()

    model_save_dir = os.path.abspath(args.exps)
    pretrained_model_file = os.path.abspath(args.model)
    data_set_dir = os.path.abspath(args.input)
    mode = args.mode

    assert mode in ['training', 'inference']

    with tf.device(args.device):
        model = modellib.MaskRCNN(mode=mode, config=config, model_dir=model_save_dir)

    model.load_weights(pretrained_model_file)

    if mode == 'training':
        dataset_train = DelftBikesDataSet()
        dataset_train.load_delft_bikes(data_set_dir, 'train')
        dataset_val = DelftBikesDataSet()
        dataset_val.load_delft_bikes(data_set_dir, 'val')

        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=1,
                    layers=args.layers)

    elif mode == 'inference':
        dataset_val = DelftBikesDataSet()
        dataset_val.load_delft_bikes(data_set_dir, 'val')

        APs = []
        for image_id in tqdm(dataset_val.image_ids):

            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset_val, config, image_id, use_mini_mask=False)

            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]

            # Compute AP
            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)

        print("mAP: ", np.mean(APs))


if __name__ == '__main__':
    args = setup_parse()
    main(args)
