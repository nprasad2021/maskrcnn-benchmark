#
# Description
#    Generated train and test sets from the Hollywood dataset byu subsampling and building image mosaics.
# 
# See:AI CAT + E2E team OneNote page for more information.  

import os, sys, shutil, xmltodict
# sys.path.append(r"C:\Users\pabuehle\Desktop\pythonLibrary")
#TODO: add path to utils
from pabuehle_utilities_CVbasic_v2 import *
from pabuehle_utilities_general_v2 import *


####################################################
# Parameters
####################################################
gridSizes = [1] #[1, 3, 5, 7]
visualize = False
numTrainImages = 50
numTestImages = 10
verbose = True

root = "/home/neeraj/Documents"
codeDir = os.path.join(root, "maskrcnn-benchmark")
rawDataDir = os.path.join(root, "data/HollywoodHeads")
outRootDir = os.path.join(root, "data/parsed")

if verbose:
      print("codeDir", codeDir)
      print("rawDataDir", rawDataDir)
      print("outRootDir", outRootDir)

imgDir = os.path.join(rawDataDir, "JPEGImages")
annoDir = os.path.join(rawDataDir, "Annotations") 
splitDir = os.path.join(rawDataDir, "Splits")

outXmlDirTrain = os.path.join(outRootDir, "train", "annotations")
outXmlDirTest  = os.path.join(outRootDir, "test", "annotations")
outImgDirTrain = os.path.join(outRootDir, "train", "images")
outImgDirTest  = os.path.join(outRootDir, "test", "images")

if verbose:
      print("outXmlDirTrain", outXmlDirTrain)
      print("outImgDirTrain", outImgDirTrain)
      print("outXmlDirTest", outXmlDirTest)
      print("outImgDirTest", outImgDirTest)


##########################
# Helpers
##########################
xmlHeader = """<?xml version="1.0" ?>
<annotation>
   <folder>Annotation</folder>
   <filename>{}</filename>      
   <path>{}</path>              
   <size>
      <width>{}</width>         
      <height>{}</height>       
      <depth>3</depth>         
   </size>
"""

xmlObject = """   <object>
      <name>head</name>           
      <pose>Unspecified</pose>
      <bndbox>
         <xmin>{}</xmin>        
         <ymin>{}</ymin>
         <xmax>{}</xmax>
         <ymax>{}</ymax>
      </bndbox>
   </object>
"""

xmlEnd = "</annotation>"



######################### 
# Main
#########################
# Visualize annotation
if visualize:
      import cvtk
      from cvtk.core import ObjectDetectionDataset
      data = ObjectDetectionDataset.create_from_dir(dataset_name='dataset', data_dir=os.path.join(outRootDir, "train"),
                                                annotations_dir="annotations", image_subdirectory='images')
      data.print_info()
      for i in range(len(data.images))[::500]:
            _ = data.images[i].visualize_bounding_boxes(image_size = (10,10))
      sdf


# Generate training and test splits 
for trainTestStr in ["test", "train"]:
      if trainTestStr == "train":
            nrImgsPerGrid = numTrainImages
            outXmlDir = outXmlDirTrain
            outImgDir = outImgDirTrain
            imgListPath = os.path.join(splitDir, "train.txt")
            imgFilenames = readFile(imgListPath) #[::5000] # 

      else:
            nrImgsPerGrid = numTestImages
            outXmlDir = outXmlDirTest
            outImgDir = outImgDirTest
            imgListPath1 = os.path.join(splitDir, "test.txt")
            imgListPath2 = os.path.join(splitDir, "val.txt")
            imgFilenames = readFile(imgListPath1)[::5] # Question
            imgFilenames += readFile(imgListPath2)

      makeDirectory(outXmlDir)
      makeDirectory(outImgDir)

      # Precompute width and height of all images
      # This is not necessary but speeds up down-stream processing.
      imgSizesFile = os.path.join(outRootDir, "imgSizes_" + trainTestStr + ".dat")
      if not os.path.exists(imgSizesFile):
            imgSizes = dict()
            for imgIndex, imgFilename in enumerate(imgFilenames):
                  imgFilename += ".jpeg"
                  if imgIndex % 100 == 0:
                        printProgressBar(float(imgIndex)/len(imgFilenames), status = "Pre-computing image width/height...")
                  imgPath = os.path.join(imgDir, imgFilename)
                  w,h = imWidthHeight(imgPath)
                  imgSizes[imgFilename] = (w,h)
            writePickle(imgSizesFile, imgSizes)
      imgSizes = readPickle(imgSizesFile)


      # Loop over image grids of 1x1, 2x2, 3x3, etc
      for gridSize in gridSizes:
            # Generate certain number of images for each grid size
            imgCount = 0
            while imgCount < nrImgsPerGrid and imgCount < len(imgFilenames):

                  # Loop over the positions in the grid
                  imgPaths = []
                  annoObjs = []
                  for gridPosW in range(gridSize):
                        for gridPosH in range(gridSize):

                              # Randomly pick images
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
                                          imgScale = max(1200 / (targetImgW * gridSize), 800 / (targetImgH * gridSize))
                                          imgScale = min(1.0, imgScale)
                                    
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

                                                # Offset and scale co-ordinates
                                                annoObj['bndbox']['xmin'] = str( (float(annoObj['bndbox']['xmin']) + offsetW) * imgScale )
                                                annoObj['bndbox']['xmax'] = str( (float(annoObj['bndbox']['xmax']) + offsetW) * imgScale )
                                                annoObj['bndbox']['ymin'] = str( (float(annoObj['bndbox']['ymin']) + offsetH) * imgScale )
                                                annoObj['bndbox']['ymax'] = str( (float(annoObj['bndbox']['ymax']) + offsetH) * imgScale )
                                                annoObjs.append(annoObj)

                                    # Add image to list
                                    boImgFound = True 
                                    imgPaths.append(imgPath)
                                    
                  # Create image mosaic
                  imgStacks = []
                  for imgPathsChunk in np.array_split(imgPaths,gridSize):
                        for imgIndex, imgPath in enumerate(imgPathsChunk):
                              img = imread(imgPath)
                              if imgIndex == 0:
                                    imgStack = imread(imgPath)
                              else:
                                    imgStack = imStack(imgStack, img)
                        imgStacks.append(imgStack)

                  outImg = []
                  for imgIndex, imgStack in enumerate(imgStacks):
                        if imgIndex == 0:
                              outImg = imgStack
                        else:
                              outImg = imConcat(outImg, imgStack)
                  outImg = imresize(outImg, imgScale)

                  # Create xml string
                  outImgFilename  = "{}_{}.jpg".format(gridSize, imgCount)
                  outAnnoFilename = "{}_{}.xml".format(gridSize, imgCount)
                  xmlStr = xmlHeader.format(outImgFilename, outImgFilename, imWidth(outImg), imHeight(outImg))
                  for annoObj in annoObjs:
                        bboxObj = annoObj['bndbox']
                        xmlStr += xmlObject.format(bboxObj['xmin'], bboxObj['ymin'], bboxObj['xmax'], bboxObj['ymax'])
                  xmlStr += xmlEnd

                  # Save    
                  outImgPath = pathJoin(outImgDir, outImgFilename)
                  outXmlPath = pathJoin(outXmlDir, outAnnoFilename)
                  imwrite(outImg, outImgPath)
                  writeFile(outXmlPath, [xmlStr])
                  imgCount += 1
                  print((gridSize, imgCount))

print("DONE.")