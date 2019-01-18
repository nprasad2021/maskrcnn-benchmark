import train_net

def main():
	parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/home/nprasad/Documents/github/maskrcnn-benchmark/configs/heads.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
	lrs = [0.1, 0.01, 0.001, 0.0001]
	for lr in lrs:
		train_net.main(args, lr)

if __name__ == "__main__":
	main()