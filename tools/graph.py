from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os, pickle, sys
print(sys.path)

from os import listdir
from os.path import isfile, join

import torch

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.engine.plotMaps import plot

def inf(args, cfg):

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    save_dir = os.path.join(cfg.OUTPUT_DIR, "testInf")
    mkdir(save_dir)
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    print(cfg.MODEL.WEIGHT)
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST) 
    dataset_names = cfg.DATASETS.TEST
    print("Dataset Names", dataset_names)
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    output_tuple = {}
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        r = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )[0].results['bbox']

        output_tuple[dataset_name] = {}
        output_tuple[dataset_name]['AP'] = r['AP'].item()
        output_tuple[dataset_name]['AP50'] = r['AP50'].item()

        synchronize()
    return output_tuple

def recordResults(args, cfg):
    homeDir = args.home
    model_paths = [cfg.MODEL.WEIGHT] + get_model_paths(join(homeDir, "output", cfg.OUTPUT_DIR, "train"))
    cfg.OUTPUT_DIR = os.path.join("output", cfg.OUTPUT_DIR, "test")
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    output = {}
    for path in model_paths:
        cfg.MODEL.WEIGHT = path
        if "final" in path:
            ite = cfg.SOLVER.MAX_ITER
        elif "no" in path:
            ite = 0
        else:
            ite = int(path.split("_")[-1].split(".")[0])
        output[ite] = inf(args, cfg)

    with open('tmp_result.pkl', 'wb') as handle:
        pickle.dump(output, handle)
    plot(output, cfg)

def get_model_paths(directory):

    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    return [join(directory, file) for file in onlyfiles if ".pth" in file]

def main():
    from maskrcnn_benchmark.config import cfg
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/home/nprasad/Documents/github/maskrcnn-benchmark/configs/heads.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--home",
        default="/home/nprasad/Documents/github/maskrcnn-benchmark",
        metavar="FILE",
        help="path to root directory",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--lr",
        default="0.1",
        help="path to config file",
        type=float,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    #cfg.SOLVER.BASE_LR = args.lr
    #cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, str(args.lr))

    recordResults(args, cfg)

if __name__ == "__main__":
    main()