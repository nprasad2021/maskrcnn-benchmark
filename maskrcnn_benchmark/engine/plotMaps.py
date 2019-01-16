import os, sys
print("System Paths:", sys.path)
import matplotlib.pyplot as plt


def plot(output, cfg):
	for mode in (cfg.DATASETS.TRAIN + cfg.DATASETS.TEST):
		runKey(mode, output, cfg.OUTPUT_DIR)

def runKey(mode, output, dirr):
	save_dir = os.path.join(dirr, mode)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	its = list(sorted(output.keys()))
	server = []
	for part in ['AP', 'AP50']:
		for i in its:
			outset = output[i][mode][part]
			server.append((i, outset))
		scatter(server, part, save_dir, mode)

def scatter(server, part, save_dir, mode):
	file_path = os.path.join(save_dir, part + "map.pdf")
	its, acc = zip(*server)
	plt.title("Accuracy on " + mode)
	plt.scatter(its,acc)
	plt.xlabel("Iterations")
	plt.ylabel(part)
	plt.savefig(file_path, dvi=1000)
