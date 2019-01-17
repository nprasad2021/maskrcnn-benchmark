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
	server = defaultdict(list)
	for part in ['AP', 'AP50']:
		for i in its:
			for mode in dsts:
				outset = output[i][mode][part]
				server[mode].append((i, outset))
		scatter(server, part, save_dir)

def scatter(server, part, save_dir):
	file_path = os.path.join(save_dir, part + "map.pdf")
	plt.title("Accuracy over Time")
	colors = ['r-', 'b', 'g']
	for i, mode in enumerate(server.keys()):
		its, acc = zip(*server[mode])
		plt.plot(its, acc, colors[(i%3)], label = mode.split("_")[1])
	plt.xlabel("Number of Training Iterations")
	plt.ylabel("Classification Accuracy")
	plt.legend()
	plt.savefig(file_path, dvi=1000)
	print("SAVED PDF OF RESULTS")
