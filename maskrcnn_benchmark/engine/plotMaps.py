import os, sys
print("System Paths:", sys.path)
import matplotlib.pyplot as plt
from collections import defaultdict


def plot(output, cfg):

	dsts = cfg.DATASETS.TEST #+ cfg.DATASETS.TRAIN
	save_dir = cfg.OUTPUT_DIR
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	its = list(sorted(output.keys()))
	print("Number of iterations saved:", len(its))
	print("Datasets", output[its[0]].keys())
	
	for part in ['AP', 'AP50']:
		server = defaultdict(list)
		for i in its:
			for mode in dsts:
				outset = output[i][mode][part]
				server[mode].append((i, outset))
		scatter(server, part, save_dir)

def scatter(server, part, save_dir):
	file_path = os.path.join(save_dir, part + "map.pdf")
	
	colors = ['r-', 'b', 'g']
	modes = list(server.keys())

	print("All Training Modes", modes)
	for i, mode in enumerate(modes):
		its, acc = zip(*(server[mode]))
		#print(its, acc, mode)
		plt.plot(its, acc, colors[i], label = mode)#.split("_")[1])
		if "test" in mode:
			test_acc = acc[-1]
		if "train" in mode:
			train_acc = acc[-1]
	plt.title(("Accuracy over Time:" + str(int(100*test_acc)) + "/" + str(int(100*train_acc))))
	plt.xlabel("Number of Training Iterations")
	plt.ylabel("Classification Accuracy")
	plt.legend()
	plt.savefig(file_path, dvi=1000)	
	plt.close()
	print("SAVED PDF OF RESULTS", file_path)

def getCGF(args):

    from maskrcnn_benchmark.config import cfg
    cfg.merge_from_file(args.config)
    cfg.OUTPUT_DIR = os.path.join("output", cfg.OUTPUT_DIR, "test")
    assert os.path.exists(cfg.OUTPUT_DIR)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
	parser.add_argument(
	    "--config",
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
	
	args = parser.parse_args()
	with open('tmp_result.pkl', 'rb') as handle:
		output = pickle.load(handle)
	cfg = getCFG(args)
	plot(output, cfg)


