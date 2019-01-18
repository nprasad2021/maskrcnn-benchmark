import os, argparse, sys
import matplotlib.pyplot as plt
from maskrcnn_benchmark.config import cfg

parser = argparse.ArgumentParser(description="Parameters for Logs Parsing")


parser.add_argument(
    "--config-file",
    default="/home/nprasad/Documents/github/maskrcnn-benchmark/configs/heads.yaml",
    metavar="FILE",
    help="path to config file",
)

args = parser.parse_args()

cfg.merge_from_file(args.config_file)
if (not "output" in cfg.OUTPUT_DIR) and (not "train" in cfg.OUTPUT_DIR):
    cfg.OUTPUT_DIR = os.path.join("output", "train", cfg.OUTPUT_DIR)
inputFileName = os.path.join(cfg.OUTPUT_DIR, "logs.txt")

def read(filename):
    if not os.path.exists(inputFileName):
        print("Input file does not exist")
        quit()
    with open(inputFileName, "r") as f:
        lines = f.readlines()
    return lines

def find_info(lines):
    tr_line_start = None
    tr_line_end = None
    l_line = None
    for i, line in enumerate(lines):
        if "maskrcnn_benchmark.trainer INFO: eta:" in line:
            tr_line_start = i
            break
    for i, line in enumerate(lines):
        if "maskrcnn_benchmark.trainer INFO: eta:" in line:
            tr_line_end = i
    for i in range(tr_line, len(lines)):
        line = lines[i]
        if "maskrcnn_benchmark.inference INFO: OrderedDict" in line:
            l_line = i
            break
    return tr_line_start, tr_line_end, l_line

def extractLoss(line):
    loss = line[line.find("loss:")+5:line.find("loss_classifier:")].split(" ")[1]
    return float(loss)
def extractIter(line):
    numIter = line[line.find("iter:")+6:line.find("iter:")+9]
    return int(numIter)

def scatter(tr_line_start, tr_line_end, save_path, title="Train Loss"):
    pairs = []
    for i in range(tr_line_start, tr_line_end+1):
        pairs.append((extractIter(lines[i]), extractLoss(lines[i])))
    iters, losses = zip(*pairs)
    plot(iters, losses, save_path, title)

def plot(x, y, save_path, title="Loss"):
    plt.title("Train Loss")
    plt.scatter(x,y)
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.savefig(save_path, dvi=1000)


def parse(filename):
    save_path = cfg.OUTPUT_DIR.replace("train", "test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    final_address = os.path.join(save_path, "train_loss.pdf")
    lines = read(filename)
    start, end, summ = find_info(lines)
    scatter(start, end, final_address)


if __name__ == "__main__":
    parse(inputFileName)





