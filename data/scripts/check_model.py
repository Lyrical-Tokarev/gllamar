"""Check trained model and view some predictions in streamlit.
Simple script: expects model name and .json file with markup, shows results for first 10 elements and metrics
"""
import click
import streamlit as st
import json

import json
import pandas as pd
import random
import cv2
import os
from tqdm import tqdm
from skimage.measure import find_contours, approximate_polygon
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
# Some basic setup:
# Setup detectron2 logger
import detectron2

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon


def get_data(json_path, subset="train"):
    if os.path.exists(json_path) and os.path.isdir(json_path):
        json_path = os.path.join(json_path, f"{subset}.json")
    with open(json_path) as f:
        data = json.load(f)
    return data

def get_mask_from_annotations(element, masks_dir="."):
    height = element['height']
    width = element['width']
    masks = np.zeros((height, width))
    for i, annotation in enumerate(element['annotations']):
        mask_path = annotation['mask_path']
        path = os.path.join(masks_dir, mask_path)
        if not os.path.exists(path):
            print(os.listdir(masks_dir))
            print(path)
        mask = cv2.imread(path, 0)/255.0
        if mask.shape != (height, width):
            #print(np.unique(mask), mask.dtype)
            resized = cv2.resize(mask, (width, height))
        else:
            resized = mask
        coords = np.argwhere(resized > 0)
        #print(resized.shape, masks.shape, (height, width))
        #print(coords[:, 0].max(), coords[:, 1].max(), height, width)
        masks[coords[:, 0], coords[:, 1]] = i+1

    return masks

@click.command()
@click.option('--model_name')
@click.option("--json_path")
@click.option("--masks_dir",
    default="raw-openimages/annotations/correct-masks"
)
def load_and_run(model_name, json_path, masks_dir):
    # cfg already contains everything we've set previously. Now we changed it a little bit for inference:
    #json_path="../annotations"
    st.text(model_name)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("alpaca_val",)
    cfg.DATASETS.TEST = ("alpaca_val", "alpaca_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR=os.path.dirname(model_name)

    cfg.MODEL.WEIGHTS = model_name
    #os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = get_data(json_path, "validation")
    st.text(json_path)
    st.text(len(dataset_dicts))
    metadata = MetadataCatalog.get("alpaca_train")
    for d in dataset_dicts:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(5,5))
        path = d["file_name"]
        #print(list(d['annotations'][0]))
        #path = os.path.join("..", file_name)
        im = cv2.imread(path)
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        ax0.imshow(out.get_image()[:, :, ::-1])
        ax0.set_title("Detectron2 prediction")
        v2 = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        out = v2.draw_dataset_dict(d)
        ax1.imshow(out.get_image()[:, :, ::-1])
        ax1.set_title("GT prediction")

        mask = get_mask_from_annotations(d, masks_dir=masks_dir)
        ax2.imshow(mask)
        ax2.set_title("mask from file")
        plt.suptitle(f"{path}")
        for ax in [ax0, ax1, ax2]:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
        st.pyplot(fig)

if __name__ == "__main__":
    load_and_run()
