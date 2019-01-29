import os, sys, argparse
from PIL import Image
import numpy as np
from predict import COCODemo
import cv2


def getCGF(args):
    from maskrcnn_benchmark.config import cfg
    cfg.merge_from_file(args.cfg)
    cfg.MODEL.WEIGHT = args.model
    assert os.path.exists(cfg.OUTPUT_DIR)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def run(args, filename):

    # prepare object that handles inference plus adds predictions on top of image
    cfg = getCGF(args)
    c = COCODemo(
        cfg,
        confidence_threshold=args.conf,
        masks_per_dim=2,
        min_image_size=224,
    )
    pil_image = Image.open(filename).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    composite = c.run_on_opencv_image(image)
    cv2.imwrite(args.save, composite)
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
        "--save",
        default="../result.jpg",
        help="path to annotated image",
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
    parser.add_argument(
        "--cfg",
        default="/home/nprasad/Documents/github/maskrcnn-benchmark/configs/heads.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--local_rank", type=int, default=0)


    args = parser.parse_args()
    print("Number of heads in image:", run(args, args.filename))

if __name__ == "__main__":
    main()