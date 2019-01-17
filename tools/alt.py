import os, sys

from os import listdir
from os.path import isfile, join

def main():
    homeDir = "/home/nprasad/Documents/github/maskrcnn-benchmark"
    cfg = getCFG()
    model_paths = [cfg.MODEL.WEIGHT] + get_model_paths(join(homeDir, cfg.OUTPUT_DIR))
    output = {}
    for path in model_paths:
    	ite, output = run(model_path)
    	output[ite] = output

    from maskrcnn_benchmark.engine.plotMaps import plot
    plot(output, cfg)

def run(model_path):
	import inf
	return inf.main(model_path)

def get_model_paths(directory):
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    return [join(directory, file) for file in onlyfiles if ".pth" in file]

def getCFG():
	from maskrcnn_benchmark.config import cfg
	cfg.merge_from_file("/home/nprasad/Documents/github/maskrcnn-benchmark/configs/heads.yaml")
	return cfg


if __name__ == "__main__":
	main()