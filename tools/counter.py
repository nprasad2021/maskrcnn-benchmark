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


def run(args):

    # prepare object that handles inference plus adds predictions on top of image
    cfg = getCGF(args)
    c = COCODemo(
        cfg,
        confidence_threshold=args.conf,
        masks_per_dim=2,
        min_image_size=224,
    )
    pil_image = Image.open(args.filename).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    composite = c.run_on_opencv_image(image)
    cv2.imwrite("../result.jpg", composite)
    return c.get_count(image)

def main():
    parser = argparse.ArgumentParser(description="Predict number of people in file")

    parser.add_argument(
        "--filename",
        default="/home/nprasad/Documents/github/testImage.jpg",
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
    print("Number of heads in image:", run(args))

if __name__ == "__main__":
    main()