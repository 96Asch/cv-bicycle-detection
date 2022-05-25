import os
import sys
import argparse

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
print(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))

from mrcnn import utils

sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN', 'samples', 'coco'))
import samples.coco as coco

def setupParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=str, help='Output folder where coco will be downloaded')   
    return parser.parse_args()

def main(argv):
    coco_path = argv.output + '/coco.h5'
    if not os.path.exists(coco_path):
        print(f'Downloading COCO weights to {coco_path}')
        utils.download_trained_weights(coco_path)
    print('Weights Downloaded')


if __name__ == '__main__':
    argv = setupParse()
    main(argv)
