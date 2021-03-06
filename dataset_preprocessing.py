# coding: utf-8
# Pipeline to convert the hollywood heads dataset to the coco format
import sys
print(sys.path)
import os, sys, shutil, xmltodict, json, argparse
from pabuehle_utilities_CVbasic_v2 import *
from pabuehle_utilities_general_v2 import *

parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")

parser.add_argument(
    "--origin",
    default="/neerajFiles/data/HollywoodHeads",
    help="path to load heads data",
    type=str,
)
parser.add_argument(
    "--output",
    default="./datasets/head/",
    help="path to dataset output",
    type=str,
)

parser.add_argument(
    "--train",
    default="400",
    help="number of training examples",
    type=int,
)

parser.add_argument(
    "--test",
    default="150",
    help="number of testing examples",
    type=int,
)

parser.add_argument(
    "--val",
    default="50",
    help="number of validation examples",
    type=int,
)

parser.add_argument(
    "--freeze",
    default=False,
    help="if true, do not replace test",
    type=bool,
)

parser.add_argument(
    "--all",
    default=False,
    help="to calculate test only if False",
    type=bool,
)

parser.add_argument(
    "--h",
    default="800",
    help="height scale",
    type=int,
)

parser.add_argument(
    "--w",
    default="1200",
    help="width scale",
    type=int,
)

parser.add_argument(
    "--m",
    default="1",
    help="min scale",
    type=int,
)

args = parser.parse_args()

subset = 0.2
gridSizes = [1, 3, 5, 7]
visualize = False
numImages = {'train':args.train, 'test':args.test, 'val':args.val}
verbose = True
heightScale = args.h
widthScale = args.w
minScale = args.m

rawDataDir = args.origin
assert os.path.exists(rawDataDir)
outDir = args.output
annotationDir = os.path.join(outDir, "annotations")

imgDir = os.path.join(rawDataDir, "JPEGImages")
annoDir = os.path.join(rawDataDir, "Annotations") 
splitDir = os.path.join(rawDataDir, "Splits")

if verbose: print(imgDir, annoDir, splitDir)
assert os.path.exists(imgDir)
assert os.path.exists(annoDir)
assert os.path.exists(splitDir)


def deleteFiles(dirPath, keywords):
    if not os.path.exists(dirPath): return
    for name in os.listdir(dirPath):
        path = os.path.join(dirPath, name)

        delete = True
        for key in keywords:
            if key in name: delete = False
        if delete:
            if os.path.isfile(path): os.remove(path)
            if os.path.isdir(path): shutil.rmtree(path)

all_stages = ['train', 'val']
if (not args.freeze) or (not os.path.exists(os.path.join(outDir, "test"))):
    all_stages.append("test")
    deleteFiles(outDir, ['dat'])
else:
    deleteFiles(annotationDir, ['test'])
    deleteFiles(outDir, ['test', 'dat', 'annotations'])

if args.all:
    all_stages.append("all")

for stage in all_stages:
    makeDirectory(os.path.join(outDir, stage))
if not os.path.exists(annotationDir):
    makeDirectory(annotationDir)



def findFilenames(stage):
    if "all" in stage:
        return findAllFilenames()
    return readFile(os.path.join(splitDir, stage + ".txt"))

def findAllFilenames():
    files = []
    for item in ['train', 'val', 'test']:
        files = files + findFilenames(item)
    return files

def computeImageSizes(stage, imgFilenamesSorted):
    imgSizesFilename = os.path.join(outDir, "imgSizes_" + stage + ".dat")
    if os.path.exists(imgSizesFilename):
        imgSizes = readPickle(imgSizesFilename)
        return imgSizes
    
    imgFilenames = sorted(imgFilenamesSorted)
    imgSizes = dict()
    
    for imgIndex, imgFilename in enumerate(imgFilenames):
        imgFilename += ".jpeg"
        if imgIndex % 5000 == 0:
            printProgressBar(float(imgIndex)/len(imgFilenames), status = "Pre-computing image width/height...")
        imgPath = os.path.join(imgDir, imgFilename)
        w,h = imWidthHeight(imgPath)
        imgSizes[imgFilename] = (w,h)
        
    writePickle(imgSizesFilename, imgSizes)
    return imgSizes

def assemble_images(gridSize, imgFilenames, imgSizes, scale=False):
    imgPaths = []
    annoObjs = []
    
    for gridPosW in range(gridSize):
        for gridPosH in range(gridSize):

            boImgFound = False
            while not boImgFound:
                imgFilename = getRandomListElement(imgFilenames) + ".jpeg"

                # Get image width and height
                imgPath = os.path.join(imgDir, imgFilename)
                w,h = imgSizes[imgFilename] #imWidthHeight(imgPath)

                # The first image specifies the width and height of all other images to be added in the grid
                if gridPosW == 0 and gridPosH == 0:
                    targetImgW = w
                    targetImgH = h
                    imgScale = max(widthScale / (targetImgW * gridSize), heightScale / (targetImgH * gridSize))
                    imgScale = min(minScale, imgScale) # 2.0 weight


                # Only add images in grid of the same size
                if w != targetImgW or h != targetImgH:
                    continue  #pick another random image until image with same w,h found

                # Load annotation
                annoFilename = imgFilename.replace(".jpeg", ".xml")
                annoPath = os.path.join(annoDir, annoFilename)
                with open(annoPath, encoding='utf-8') as f:
                    xmlData = xmltodict.parse(f.read())
                if type(xmlData['annotation']['object']) != list:
                    xmlData['annotation']['object'] = [ xmlData['annotation']['object'] ]

                # Offset to transform ground truth annotations into the image grid
                offsetW = gridPosW * targetImgW
                offsetH = gridPosH * targetImgH

                # Loop over all annotations
                for annoObj in xmlData['annotation']['object']:
                    if xmlData['annotation']['object'] != [None]:
                        assert(annoObj['name'] == 'head')
                        annoObj['bndbox']['xmin'] = (float(annoObj['bndbox']['xmin']) + offsetW) * imgScale
                        annoObj['bndbox']['xmax'] = (float(annoObj['bndbox']['xmax']) + offsetW) * imgScale
                        annoObj['bndbox']['ymin'] = (float(annoObj['bndbox']['ymin']) + offsetH) * imgScale
                        annoObj['bndbox']['ymax'] = (float(annoObj['bndbox']['ymax']) + offsetH) * imgScale
                        annoObjs.append(annoObj)

                # Add image to list
                boImgFound = True 
                imgPaths.append(imgPath)
                
    assert len(imgPaths) == gridSize**2
    return imgPaths, annoObjs, imgScale

def image_mosaic(gridSize, imgPaths, imgScale):

    imgStacks = []
    
    for imgPathsChunk in np.array_split(imgPaths,gridSize):
        for imgIndex, imgPath in enumerate(imgPathsChunk):
            img = imread(imgPath)
            if imgIndex == 0:
                imgStack = imread(imgPath)
            else:
                imgStack = imStack(imgStack, img)
        imgStacks.append(imgStack)
        
    for imgIndex, imgStack in enumerate(imgStacks):
        if imgIndex == 0:
            outImg = imgStack
        else:
            outImg = imConcat(outImg, imgStack)
            
    outImg = imresize(outImg, imgScale)
    return outImg

def calculate_coco_bounding_box(bbox, width, height):
    return float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']-bbox['xmin']), float(bbox['ymax']-bbox['ymin'])

def annotate_image(gridSize, imgFilenames, imgSizes, imgCount, annotationCount, stage, store_sub=False):
    imgPaths, annoObjs, imgScale = assemble_images(gridSize, imgFilenames, imgSizes)
    
    outImg = image_mosaic(gridSize, imgPaths, imgScale)
    width, height = imWidthHeight(outImg)
    
    outImgFilename  = "{}_{}.jpg".format(gridSize, imgCount)
    outImgPath = os.path.join(outDir, stage, outImgFilename)
    print(outImgPath)
    imwrite(outImg, outImgPath)

    if store_sub:
        subFolder = os.path.join(outDir, "sub")
        if not os.path.exists(subFolder):
            makeDirectory(subFolder)
        outSubPath = os.path.join(outDir, "sub", outImgFilename)
        imwrite(outImg, outSubPath)
    
    image_annotation = {"id":imgCount, "width":width, "height":height, "license":1, "file_name":outImgFilename}
    object_annotations = []
    
    for annoObj in annoObjs:
        annotationCount += 1
        
        x, y, oW, oH = calculate_coco_bounding_box(annoObj['bndbox'], width, height)
        nextA = {"id":annotationCount, "image_id":imgCount, "category_id":1, "iscrowd":0, "bbox":[x,y,oW,oH], 'area':width*height}
        object_annotations.append(nextA)
    return image_annotation, object_annotations, annotationCount


def dump_annotations(image_annotations, object_annotations, stage):
    info = {"year":2019, "version":1, "description": "Hollywood Heads", "contributor": "Neeraj, Patrick"}
    license = {"id":1, "name":"MIT", "url":"www.google.com"}
    categories = [{"id":1, "name":"head", "supercategory":"header"}]
    final = {"categories":categories, "images":image_annotations, "annotations":object_annotations, 
                "info": info, "license":license}
    outpath = os.path.join(annotationDir, "instances_" + stage + ".json")
    with open(outpath, "w") as fp:
        json.dump(final, fp)

def main():
    imgCount = 0
    annotationCount = 0
    
    for stage in all_stages:
        image_annotations = []
        object_annotations = []
        
        imgFilenames = findFilenames(stage)
        imgSizes = computeImageSizes(stage, imgFilenames)
        sub = False
        if "all" in stage:
            numImages[stage] = len(imgFilenames)

        if "train" in stage: 
            sub = True
            image_annotations_sub = []
            object_annotations_sub = []

        for gridSize in gridSizes:
            under = True
            for i in range(numImages[stage] // len(gridSizes)):
                imgCount += 1

                if i > (numImages[stage]*subset): under = False

                im_anno, obj_anno, annotationCount = annotate_image(gridSize, imgFilenames, imgSizes, imgCount, 
                                                                        annotationCount, stage, (sub and under))
                
                image_annotations.append(im_anno)
                object_annotations.extend(obj_anno)
                if sub and under:
                    image_annotations_sub.append(im_anno)
                    object_annotations_sub.extend(obj_anno) 

                
        dump_annotations(image_annotations, object_annotations, stage)
        if sub:
            dump_annotations(image_annotations_sub, object_annotations_sub, "sub")
        
if __name__ == "__main__":
    main()

