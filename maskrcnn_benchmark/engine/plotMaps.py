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
	plt.title("Accuracy over Time")
	colors = ['r-', 'b', 'g']
	modes = list(server.keys())

	print("All Training Modes", modes)
	for i, mode in enumerate(modes):
		its, acc = zip(*(server[mode]))
		#print(its, acc, mode)
		plt.plot(its, acc, colors[i], label = mode)#.split("_")[1])
	plt.xlabel("Number of Training Iterations")
	plt.ylabel("Classification Accuracy")
	plt.legend()
	plt.savefig(file_path, dvi=1000)
	plt.close()
	print("SAVED PDF OF RESULTS")
