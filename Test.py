import numpy as np
from scipy import ndimage
from os import listdir
import matplotlib.pyplot as plt
from PIL import Image

# ******************    Connected Component Analysis
# testMatrix = np.array([[[1, 1, 0],
#                         [0, 0, 0],
#                         [1, 0, 0]],
#                        [[0, 1, 0],
#                         [0, 0, 0],
#                         [1, 0, 0]],
#                        [[0, 1, 1],
#                         [0, 0, 0],
#                         [1, 0, 1]]])
#
# outMatrix = ndimage.label(testMatrix)
#
# matrix = outMatrix[0]
#
# print(matrix)
# print('***')
#
# matrix[matrix != 2] = 0
# matrix[matrix == 2] = 1
#
# print(matrix)

# ******************   Erosion Operation
# testMatrix = np.array([[[1, 0, 0, 0, 1],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [1, 0, 0, 0, 1]],
#                        [[0, 1, 0, 1, 1],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 0]],
#                        [[0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 1],
#                         [0, 1, 1, 1, 1],
#                         [0, 1, 1, 1, 1],
#                         [0, 0, 1, 1, 1]]])
#
# kernel = np.ones((2,2,2))
#
# print(kernel)
#
# outputMatrix = ndimage.binary_erosion(testMatrix, structure=kernel).astype(testMatrix.dtype)
#
# print(outputMatrix)

# ************* Dilation Operration
# testMatrix = np.array([[[0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 0]],
#                        [[0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 0]],
#                        [[0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 1]]])
#
# kernel = np.ones((2,2,2))
#
# print(kernel)
#
# outputMatrix = ndimage.binary_dilation(testMatrix, structure=kernel).astype(testMatrix.dtype)
#
# print(outputMatrix)
# kernel = ndimage.generate_binary_structure(3,1)
# print(kernel)

# ************  Get File List
# fileNames = listdir(r'C:\Users\polat\Desktop\TEZ\LUNA\subset0')
#
# print(fileNames[1])

# ************ Image Show
# image = np.array([[0.1, 0.2, 0.7, 0.5, 0.9],
#                   [0.2, 0.4, 0.5, 0.5, 0.7],
#                   [0.2, 0.3, 0.1, 0.5, 0.7],
#                   [0.7, 0.7, 0.1, 0.5, 0.3],
#                   [0.9, 0.7, 0.9, 0.3, 0.1]])
#
# img = Image.fromarray(np.uint8(255 * image))
# img.show()

# ************* Save Numpy Array
# testMatrix = np.array([[[0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 0]],
#                        [[0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 0]],
#                        [[0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 1]]])
#
# np.save('3darr',testMatrix)

# *************** Read Numpy Array
# readArr = np.load('3darr.npy')
#
# print(readArr)

# *************** Resize 3D volume
# WIDTH = 6
# ZWIDTH = 4
# testMatrix = np.array([[[0, 0, 0, 0, 0],
#                         [0, 0.5, 1, 0.5, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0.5, 1,0.5, 0],
#                         [0, 0, 0, 0, 0]],
#                        [[0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 0]],
#                        [[0, 0, 0, 0, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 1, 1, 1, 0],
#                         [0, 0, 0, 0, 1]]])
#
# zoomFactor = [ZWIDTH / float(testMatrix.shape[0]), WIDTH / float(testMatrix.shape[1]),
#               WIDTH / float(testMatrix.shape[2])]
# resizedImage = ndimage.zoom(testMatrix, zoom=zoomFactor)
#
# print(resizedImage)
# print(resizedImage.shape)

testMatrix = np.array([0,1.0,2.0,1.0,0])

resizedImage = ndimage.zoom(testMatrix,zoom=2, mode='nearest')

print(resizedImage)