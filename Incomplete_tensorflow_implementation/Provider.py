import SimpleITK as sitk
import numpy as np
import csv
import tensorflow as tf

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)

    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def readCSV(filename):
    lines = []
    with open(filename,"r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def worldToVoxelCoord(worldCoord, origin, spacing):
    strechedVoxelCoord = np.absolute(worldCoord-origin)
    voxelCoord = strechedVoxelCoord / spacing
    return voxelCoord

def normalizePlanes(npzarray):

    maxHU = 400.
   # maxHU = np.amax(npzarray)
    minHU = -1000.
   # minHU = np.amin(npzarray)

    npzarray = (npzarray-minHU) / (maxHU-minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.

    return  npzarray

def mixArrays(firstArray,secondArray):
    resultList  = []
    for i in range(2*len(firstArray)):
        if i%2 == 0:
            resultList.append(firstArray)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def makeOutputArray(output):
    if (output == 0):
        return np.array([1,0])
    else:
        return np.array([0, 1])