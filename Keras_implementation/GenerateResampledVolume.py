from os import listdir
import numpy as np
import Provider
from scipy import ndimage

# Directories
baseSubsetDirectory = r'/media/gorkem/TI31299000D/LUNA DATA/subsets/subset'
targetSubsetDirBase = r'/media/gorkem/TI31299000D/LUNA DATA/Resampled/subsets/subset'

RESIZE_SPACING = [1, 0.7, 0.7]
NUM_SUBSET = 10

for setNumber in range(NUM_SUBSET):
    subsetDirectory = baseSubsetDirectory + str(setNumber)
    list = listdir(subsetDirectory)
    subsetList = []

    # Create Subset List
    for file in list:
        if file.endswith(".mhd"):
            subsetList.append(file)

    for file in subsetList:
        try:
            fileName = file[:-4]
            filePath = subsetDirectory + '/' + file
            volumeImage, numpyOrigin, numpySpacing = Provider.load_itk_image(filePath)

            resize_factor = numpySpacing / RESIZE_SPACING
            new_real_shape = volumeImage.shape * resize_factor
            new_shape = np.round(new_real_shape)
            real_resize = new_shape / volumeImage.shape

            new_volume = ndimage.zoom(volumeImage, zoom=real_resize)

            np.save(targetSubsetDirBase + str(setNumber) + '/' + fileName, new_volume)

            logFile = open('log_GenerateResampledVolume.txt', 'a')
            logFile.write(str(setNumber) + ' -> ' + fileName + '\n')
            logFile.close()
        except:
            logFile = open('log_GenerateResampledVolume.txt', 'a')
            logFile.write('ERROR: '+str(setNumber) + ' -> ' + fileName + '\n')
            logFile.close()
