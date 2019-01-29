# How to train and test the head detector

## Setup and Installation

Run the following instructions in your shell

```sh
mkdir head_detector
cd head_detector
git clone git@github.com:nprasad2021/maskrcnn-benchmark.git
cd maskrcnn-benchmark
python3 -m venv ./mask
source ./mask/bin/activate
pip3 install ninja yacs cython matplotlib
pip3 install opencv-python==3.4.5.20
pip3 install torch torchvision
cd ..
git clone git@github.com:cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cd ../../maskrcnn-benchmark
python3 setup.py build develop
cd ..
wget -P data http://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip
unzip data/HollywoodHeads.zip -d data
cd maskrcnn-benchmark
```

## Dataset Preprocessing
Navigate to the maskrcnn-benchmark directory. As the script currently stands,
you will have to manually change data augmentation parameters.
```python
python dataset_preprocessing.py --origin ../data/HollywoodHeads
    --train NUM_TRAIN_IMAGES --test NUM_TEST_IMAGES --val NUM_VAL_IMAGES
```


