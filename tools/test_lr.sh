#!/bin/bash
python tools/graph.py --lr 0.1
wait
python tools/graph.py --lr 0.01
wait
python tools/graph.py --lr 0.001
wait
python tools/graph.py --lr 0.0001