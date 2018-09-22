import SimpleITK as sitk
import numpy as np
import csv
import tensorflow as tf
from datetime import datetime

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)

    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


def worldToVoxelCoord(worldCoord, origin, spacing):
    strechedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = strechedVoxelCoord / spacing
    return voxelCoord


def normalizePlanes(npzarray):
    maxHU = 400.
    # maxHU = np.amax(npzarray)
    minHU = -1000.
    # minHU = np.amin(npzarray)

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.

    return npzarray

def writeToDebugFile(message):
    logFile = open('log_main.txt', 'a')
    logFile.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' -> '+ message + '\n')
    logFile.close()

def mixArrays(firstArray, secondArray):
    resultList = []
    for i in range(2 * len(firstArray)):
        if i % 2 == 0:
            resultList.append(firstArray[int(i / 2)])
        else:
            resultList.append(secondArray[int(i / 2)])
    return resultList

def mixArraysNumpy(firstArray, secondArray):
    for i in range(2 * len(firstArray)):
        if (i == 0):
            tempArray = firstArray[0]
            result = np.array(tempArray[np.newaxis,...])
        elif i % 2 == 0:
            tempArray = firstArray[int(i / 2)]
            result = np.concatenate((result, tempArray[np.newaxis,...]), axis=0)
        else:
            tempArray = secondArray[int(i / 2)]
            result = np.concatenate((result, tempArray[np.newaxis,...]), axis=0)
    return result

def mixArraysNumpy2(firstArray, secondArray, zDimension, xyDimension):
    result = np.empty([2*len(firstArray), zDimension, xyDimension, xyDimension])
    for i in range(2 * len(firstArray)):
        if i % 2 == 0:
            result[i] = firstArray[int(i / 2)]
        else:
            result[i] = secondArray[int(i / 2)]
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def makeOutputArray(output):
    if (output == 0):
        return np.array([1, 0])
    else:
        return np.array([0, 1])
