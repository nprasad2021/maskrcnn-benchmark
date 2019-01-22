#!/bin/bash
python train_net.py --lr 0.1
wait 
python train_net.py --lr 0.01
wait
python train_net.py --lr 0.001
wait
python train_net.py --lr 0.0001