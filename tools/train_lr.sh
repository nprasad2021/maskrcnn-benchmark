#!/bin/bash
python tools/train_net.py --lr 0.1
wait 
python tools/train_net.py --lr 0.01
wait
python tools/train_net.py --lr 0.001
wait
python tools/train_net.py --lr 0.0001