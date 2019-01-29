# How to train and test the head detector
Neeraj Prasad
## Setup and Installation

Run the following instructions in bash in your Linux Environment with GPUs

```bash
git clone git@gist.github.com:9c3d406f69580929fe86023154dd4390.git
mv 9c3d406f69580929fe86023154dd4390/head_detector.sh ./head_detector.sh
chmod +x head_detector.sh
./head_detector.sh
```

## Dataset Preprocessing
Navigate to the ```head_detector/maskrcnn-benchmark``` directory. As the script currently stands,
you will have to manually change data augmentation parameters if you wish to add augmentation.
```
python dataset_preprocessing.py --origin ../data/HollywoodHeads
    --train NUM_TRAIN_IMAGES --test NUM_TEST_IMAGES --val NUM_VAL_IMAGES
```

If you wish to not modify the test dataset, set the parameter --freeze True

## Pretrained Model Initialization

Do you want to use a model pre-trained on ImageNet or on Coco?

### ImageNet

Navigate to the configs directory. The file ```heads.yaml``` is the configuration you should edit. Choose 
a pre-trained ImageNet model following the example from the other config files in your directory.
Change the ```MODEL.WEIGHT``` parameter to the URL for the ImageNet model.

### Coco

Choose a model that you wish to use from the [model zoo](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/MODEL_ZOO.md)
Once you have selected the model, select copy link address from the hyperlink
under the field ```model_id``` and run the following command in the 
maskrcnn-benchmark directory:

```
python trim_detectron_model.py --cfg [PATH_TO_CONFIG_FILE]
```

If you do not wish to use the default model, you can add the optional argument
```--url``` and select one of the following:
 ```
"35857197/e2e_faster_rcnn_R-50-C4_1x",
"35857345/e2e_faster_rcnn_R-50-FPN_1x"
"35857890/e2e_faster_rcnn_R-101-FPN_1x"
"36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x"
"35858791/e2e_mask_rcnn_R-50-C4_1x"
"35858933/e2e_mask_rcnn_R-50-FPN_1x"
"35861795/e2e_mask_rcnn_R-101-FPN_1x"
"36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x"
```
For example, to use ResNet-101

``` 
python trim_detectron_model.py --url 35857890/e2e_faster_rcnn_R-101-FPN_1x --cfg PATH_TO_CONFIG
```

You may also specify the name of file you wish to save with --save-file.
Now, modify the ```MODEL.WEIGHT``` parameters in your config file such that it
points to the model you saved.

## Run

How to train and test your model.

### Config File

You may have to modify ```MODEL.BACKBONE.CONV_BODY``` based on the 
pretrained network you choose to use. ```SOLVER``` has all relevant hyper-parameters.
Select your ```output_dir``` based on the title of your experiment. Note that all training
and testing results for your experiments will be stored in the directory ```output/[output_dir]/train```
and ```output/[output_dir]/test``` respectively in the ```maskrcnn-benchmark``` repo.

### Train

To train your model, use the following command:
```
python tools/train_net.py --config-file [PATH_TO_CONFIG] --skip-test [bool]
```
Note that all intermediate models will be stored in ```output/[output_dir]/train```


### Validate

To generate accuracies and plots for your model, use the following command:

```
python tools/graph.py --config-file [PATH_TO_CONFIG]
```

Note that all results will be stored in ```output/[output_dir]/test```.


### Test

To count the number of heads in an user-picked image, run the following command:

```
python tools/counter.py --filename [PATH_TO_IMAGE_FILE] --model [PATH_TO_MODEL] \
    --output [PATH_TO_ANNOTATED_IMAGE] --conf [CONFIDENCE]
```

To convert a video file to an annotated video for head detection:

```
python demo/video.py --config-file [CONFIG_FILE_PATH] --modelpath [MODEL_PATH] \
    --savepath [ANNOTATED_VIDEO_PATH] --input [INPUT_VIDEO_PATH] --confidence-threshold [CONF]
```

To run the web-cam, real-time with object detection:

```sh
python demo/webcam.py --config-file [CONFIG_FILE_PATH] --modelpath [PATH_TO_MODEL] # on GPU
python demo/webcam.py --min-image-size 300 MODEL.DEVICE cpu # on CPU (still need modelpath and config params)
```






