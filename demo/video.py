# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time
import sys

print(sys.path)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/heads.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--modelpath",
        default="../trained_model.pth",
        help="path to model",
    )

    parser.add_argument(
        "--savepath",
        default="demo.avi",
        help="save path",
    )

    parser.add_argument(
        "--input",
        help="save path",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
             "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.modelpath
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    cam = cv2.VideoCapture(args.input)
    frame_width = int(cam.get(3));
    frame_height = int(cam.get(4))
    out = cv2.VideoWriter(args.savepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    if (cam.isOpened() == False):
        print("Error opening video stream or file")
    else:
        print("no error opening video stream")
    ret_val, img = cam.read()
    while ret_val:
        start_time = time.time()
        composite = coco_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        # cv2.imshow("COCO detections", composite)
        out.write(composite)
        ret_val, img = cam.read()

    cam.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
