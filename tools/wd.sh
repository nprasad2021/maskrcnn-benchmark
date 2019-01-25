#!/bin/sh
python tools/train_net.py --wd 0.0001
wait
python tools/train_net.py --wd 0.0005
wait
python tools/train_net.py --wd 0.001
wait
python tools/graph.py --wd 0.0001
wait
python tools/graph.py --wd 0.0005
wait
python tools/graph.py --wd 0.001