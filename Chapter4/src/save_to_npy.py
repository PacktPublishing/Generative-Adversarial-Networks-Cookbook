#!/usr/bin/env python3
from PIL import Image
import numpy as np
import os


def grabListOfFiles(startingDirectory,extension=".webp"):
    listOfFiles = []
    for file in os.listdir(startingDirectory):
        if file.endswith(extension):
            listOfFiles.append(os.path.join(startingDirectory, file))
    return listOfFiles

def grabArrayOfImages(listOfFiles,resizeW=64,resizeH=64,gray=False):
    imageArr = []
    for f in listOfFiles:
        if gray:
            im = Image.open(f).convert("L")
        else:
            im = Image.open(f).convert("RGB")
        im = im.resize((resizeW,resizeH))
        imData = np.asarray(im)
        imageArr.append(imData)
    return imageArr

direc = "/data/church_outdoor_train_lmdb/expanded/"
listOfFiles = grabListOfFiles(direc)
imageArrGray = grabArrayOfImages(listOfFiles,resizeW=64,resizeH=64,gray=True)
imageArrColor = grabArrayOfImages(listOfFiles,resizeW=64,resizeH=64)
print("Shape of ImageArr Gray: ", np.shape(imageArrGray))
print("Shape of ImageArr Color: ", np.shape(imageArrColor))
np.save('/data/church_outdoor_train_lmdb_gray.npy', imageArrGray)
np.save('/data/church_outdoor_train_lmdb_color.npy', imageArrColor)

