import click
import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultTrainer
# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader

from dummy_albu_mapper import DummyAlbuMapper


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DummyAlbuMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)


def get_data(json_path, subset="train"):
    if os.path.exists(json_path) and os.path.isdir(json_path):
        json_path = os.path.join(json_path, f"{subset}.json")
    with open(json_path) as f:
        data = json.load(f)
    return data

@click.command()
@click.argument("json_path")
def train(json_path):
    for d in ["train", "test", "validation"]:
        DatasetCatalog.register("alpaca_" + d, lambda d=d: get_data(json_path,  d))
        MetadataCatalog.get("alpaca_" + d).set(thing_classes=["alpaca"])
    balloon_metadata = MetadataCatalog.get("alpaca_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("alpaca_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.ALBUMENTATIONS = './scripts/sample-detectron-albu-config.json'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    train()
