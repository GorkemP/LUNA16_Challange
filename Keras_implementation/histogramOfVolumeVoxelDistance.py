from os import listdir
import numpy as np
import Provider
import matplotlib.pyplot as plt

# Directories
baseSubsetDirectory = r'/media/gorkem/TI31299000D/LUNA DATA/subsets/subset'

NUM_SUBSET = 10

XYList = []
ZList = []
for setNumber in range(NUM_SUBSET):
    subsetDirectory = baseSubsetDirectory + str(setNumber)
    list = listdir(subsetDirectory)
    subsetList = []

    # Create Subset List
    for file in list:
        if file.endswith(".mhd"):
            subsetList.append(file)

    for file in subsetList:
        fileName = file[:-4]
        filePath = subsetDirectory + '/' + file
        volumeImage, numpyOrigin, numpySpacing = Provider.load_itk_image(filePath)

        ZList.append(numpySpacing[0])
        XYList.append(numpySpacing[2])
	print(numpSpacing)

thefile = open('XYDims.txt', 'w')
for item in XYList:
    thefile.write('%s\n' % item)

thefile = open('ZDims.txt', 'w')
for item in ZList:
    thefile.write('%s\n' % item)

plt.hist(XYList, bins=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1])
plt.title("Histogram of Voxel Distances in Transverse Plane")
plt.xlabel('Distance in mm')
plt.ylabel('Frequency')
plt.savefig('XYList.png')

plt.hist(ZList, bins=[0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5])
plt.title("Histogram of Distances between Slices")
plt.xlabel('Distance in mm')
plt.ylabel('Frequency')
plt.savefig('ZList.png')