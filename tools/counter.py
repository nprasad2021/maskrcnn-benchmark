import os, sys, argparse
from PIL import Image
import numpy as np
from predict import COCODemo


def getCGF():
    cfg.merge_from_file("/home/nprasad/Documents/github/maskrcnn-benchmark/configs/heads.yaml")
    cfg.OUTPUT_DIR = os.path.join("output", "train", cfg.OUTPUT_DIR)
    assert os.path.exists(cfg.OUTPUT_DIR)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def run(image_filename):

    # prepare object that handles inference plus adds predictions on top of image
    cfg = getCGF()
    c = COCODemo(
        cfg,
        confidence_threshold=0.7,
        masks_per_dim=2,
        min_image_size=224,
    )
    pil_image = Image.open(image_filename).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return c.get_count(image) 