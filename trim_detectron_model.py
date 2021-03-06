import os, torch, argparse, urllib.request
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from pabuehle_utilities_CVbasic_v2 import *
from pabuehle_utilities_general_v2 import *
from maskrcnn_benchmark.config.paths_catalog import ModelCatalog

def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r

parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")

parser.add_argument(
    "--save_file",
    default="fastercnnmodel_no_last_layers.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument( 
    "--cfg",
    default="configs/heads.yaml",
    help="path to config file",
    type=str,
)

parser.add_argument(
    "--url",
    default="Caffe2Detectron/COCO/35857345/e2e_faster_rcnn_R-50-FPN_1x",
    help="url to file",
    type=str,
)

args = parser.parse_args()

pretrained_path = "../tmp.pth"
URL = ModelCatalog.get(args.url)
urllib.request.urlretrieve(URL, pretrained_path)

saveDir = "../pretrained_models"
makeDirectory(saveDir)

DETECTRON_PATH = os.path.expanduser(pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d

newdict['model'] = removekey(_d['model'],
                             ['cls_score.bias', 'cls_score.weight', 'bbox_pred.bias', 'bbox_pred.weight'])
torch.save(newdict, os.path.join(saveDir, args.save_file))
print('saved to {}.'.format(os.path.join(saveDir, args.save_file)))
