# Hard Negative Mining

import os, sys, argparse
from PIL import Image
import numpy as np
from predict import COCODemo
import cv2


def getCGF(args):
    from maskrcnn_benchmark.config import cfg
    cfg.merge_from_file("/home/nprasad/Documents/github/maskrcnn-benchmark/configs/heads.yaml")
    cfg.OUTPUT_DIR = args.output
    cfg.MODEL.WEIGHT = args.model
    assert os.path.exists(cfg.OUTPUT_DIR)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def run(args, filepaths, save_path="../result.jpg"):

    # prepare object that handles inference plus adds predictions on top of image
    cfg = getCGF(args)
    c = COCODemo(
        cfg,
        confidence_threshold=args.conf,
        masks_per_dim=2,
        min_image_size=224,
    )
    for filepath in filepaths:
    	filename = filepath.split('/')[-1]
	    pil_image = Image.open(filepath).convert("RGB")
	    image = np.array(pil_image)[:, :, [2, 1, 0]]
	    composite = c.run_on_opencv_image(image)
	    cv2.imwrite(os.path.join(save_path, filename), composite)


def main(args):
	counter.run(args, )



def parse_args():
	parser = argparse.ArgumentParser(description="Hard Negative Mining")
	parser.add_argument(
	    "--dir",
	    default="/home/nprasad/Documents/github/maskrcnn-benchmark/datasets/head/",
	    help="path to test image",
	    type=str,
	)
	parser.add_argument(
	    "--model",
	    default="/home/nprasad/Documents/github/maskrcnn-benchmark/output/lr/0.0001/train/model_0004400.pth",
	    help="path to dataset output",
	    type=str,
	)
	parser.add_argument(
	    "--output",
	    default=os.path.join("output", "lr/0.0001/test"),
	    help="path to dataset output",
	    type=str,
	)
	parser.add_argument(
	    "--conf",
	    default=0.3,
	    type=float,
	)
	parser.add_argument(
	    "opts",
	    help="Modify config options using the command-line",
	    default=None,
	    nargs=argparse.REMAINDER,
	)
	args = parser.parse_args()
	return args



if __name__ == "__main__":

    args = parse_args()
    print("Number of heads in image:", run(args))

