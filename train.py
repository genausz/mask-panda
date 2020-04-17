import os
#import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.data.datasets import register_coco_instances
register_coco_instances("panda_train", {}, "panda_coco/annotations/panda_train.json", "panda_coco/train")
register_coco_instances("panda_val", {}, "panda_coco/annotations/panda_val.json", "panda_coco/val")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
cfg.DATASETS.TRAIN = ("panda_train",)
cfg.DATASETS.TEST = ("panda_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")  # Let training initialize from model zoo
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1               # IMPORTANT HERE!  1 for panda
cfg.MODEL.ROI_HEADS.BATCH_SIZE_FOR_IMAGE = 512

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025   # pick a good LR
cfg.SOLVER.WARMUP_ITERS = 200  # warmup lr before iter < 100
cfg.SOLVER.GAMMA = 0.3
cfg.SOLVER.STEPS = (450, 700)
cfg.SOLVER.MAX_ITER = 800     # 500 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.INPUT.MAX_SIZE_TEST = 1280
cfg.INPUT.MAX_SIZE_TRAIN = 1280
print(cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
