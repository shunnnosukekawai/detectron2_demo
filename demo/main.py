# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

from detectron2.data import MetadataCatalog
MetadataCatalog.get("dla_val").thing_classes = ['text', 'title', 'list', 'table', 'figure']
# constants
WINDOW_NAME = "COCO detections"

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file('configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml')
    #change the model path
    path = './model/'
    files = os.listdir(path)
    file_path = path + files[1]
    cfg.merge_from_list(['MODEL.WEIGHTS', file_path, 'MODEL.DEVICE', 'cpu'])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg

import click
@click.command()
@click.option('-i', help='input file path')
@click.option('-o', help='output file path')

def cmd(i,o):
    mp.set_start_method("spawn", force=True)
    cfg = setup_cfg()
    demo = VisualizationDemo(cfg)
    img = read_image(i, format="BGR")
    inputimg_shape = cv2.imread(i).shape[:2]
    predictions, visualized_output = demo.run_on_image(img)
    visualized_output.save(o)

    #output prediction results to json
    import json
    prediction_dict = predictions['instances'].get_fields()
    classes = ['text', 'title', 'list', 'table', 'figure']

    prediction_result = {}
    prediction_result['input_img shape'] = inputimg_shape
    for k in prediction_dict:
        if(k == 'pred_boxes'):
            prediction_result['detected box areas'] = prediction_dict[k].tensor.tolist()
            big_box = []
            for i in prediction_dict[k].tensor.tolist():
                small_box = []
                for j in range(len(i)):
                    if((j == 0) or (j == 2)):
                        small_box.append(i[j]/inputimg_shape[1])
                    elif((j == 1) or (j == 3)):
                        small_box.append(i[j]/inputimg_shape[0])
                big_box.append(small_box)
            prediction_result['overall ratio of detected boxes'] = big_box
        elif(k == 'scores'):
            prediction_result['confidence scores']  = prediction_dict[k].tolist()
        elif(k == 'pred_classes'):
            categories = []
            for i in prediction_dict[k].tolist():
                categories.append(classes[i])
            prediction_result['categories'] = categories
    f = open('prediction_result.json', 'w')
    json.dump(prediction_result, f)
    print('successed')

if __name__ == "__main__":
    cmd()
