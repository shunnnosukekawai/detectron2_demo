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

    #make categories list
    categories = []
    for i in prediction_dict['pred_classes'].tolist():
        categories.append(classes[i])

    #make overall ratio list
    overall_ratio = []
    for i in prediction_dict['pred_boxes'].tensor.tolist():
        small_box = []
        for j in range(len(i)):
            if((j == 0) or (j == 2)):
                small_box.append(i[j]/inputimg_shape[1])
            elif((j == 1) or (j == 3)):
                small_box.append(i[j]/inputimg_shape[0])
        overall_ratio.append(small_box)

    detected_boxes_list = []
    for i in range(len(prediction_dict['pred_boxes'])):
        detected_bos_and_category = {}
        detected_bos_and_category['categorie'] = categories[i]
        detected_bos_and_category['detected box area'] = prediction_dict['pred_boxes'].tensor.tolist()[i]
        detected_bos_and_category['overall ratio of detected box'] = overall_ratio[i]
        detected_bos_and_category['confidence score'] = prediction_dict['scores'].tolist()[i]
        detected_boxes_list.append(detected_bos_and_category)

    prediction_result['detected boxes'] = detected_boxes_list
    f = open('prediction_result.json', 'w')
    json.dump(prediction_result, f)
    print('successed')

if __name__ == "__main__":
    cmd()
