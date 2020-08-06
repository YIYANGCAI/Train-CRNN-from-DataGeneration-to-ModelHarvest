# -*- coding: utf-8 -*-
import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob

def judgeStr(fatherStr, sonStr):
    # fatherStr is a set which contains all chars
    # if some char is not included in the fatherStr
    # they will be add in fatherStr
    for ch in sonStr:
        if ch not in fatherStr:
            fatherStr.append(ch)
            print("current alphabets:\t{}".format(fatherStr))
    return fatherStr

def findClasses(label_files, icdarType = True):
    alphabets = []
    f = open(label_files)
    # all samples -> get a alphabet contains all chars in the datasets
    for line in list(f):
        #print("Loading the file dual:\t{}".format(line.split()[0]))
        label = line.split()[1]
        if icdarType:
            # To segment the original label from head-tail syntax
            label = label[1:-1]
        alphabets = judgeStr(alphabets, label)
        #imagePathList.append('foo_datum/' + image+".jpg")
    return alphabets

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print('...................')
    # map_size=1099511627776 定义最大空间是1TB
    env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        #print imagePath
        #imagePath = imagePath.replace(' ','')
        label = labelList[i]
        # delete the space in the image path
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label

        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def read_text(path):
    with open(path) as f:
        text = f.read()
    text = text.strip()

    return text

if __name__ == '__main__':

    outputPath = './datasets/second_data/test_lmdb/'
    imgdata = open("./datasets/second_data/test.txt")
    imagePathList = []
    imgLabelLists = []
    for line in list(imgdata):
        label = line.split(' ')[1][0:-1] # delete the \n in the tail
        # Typically for the usage of the icdar-2015 datasets
        #label = label[1:-1]
        #image = line.split()[0][0:-1]
        image = line.split(' ')[0]
        #print image
        imgLabelLists.append(label)
        imagePathList.append(os.path.join('./datasets/second_data/test/', image))

    print len(imagePathList)
    print len(imgLabelLists)
    print imagePathList[222]
    print imgLabelLists[222]
    createDataset(outputPath, imagePathList, imgLabelLists, lexiconList=None, checkValid=True)