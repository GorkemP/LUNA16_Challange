from os import listdir
import numpy as np
import Provider
from scipy import ndimage
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.callbacks import EarlyStopping
from keras import backend as K
from Provider import writeToDebugFile

writeToDebugFile('Program Started')

# Directories
originalSubsetDirectoryBase =  r'/media/gorkem/TI31299000D/LUNA DATA/subsets/subset'
resampledSubsetDirectoryBase = r'/media/gorkem/TI31299000D/LUNA DATA/Resampled/subsets/subset'
candidates =                   r'/media/gorkem/TI31299000D/LUNA DATA/CSVFILES/candidates_V2.csv'

RESIZE_SPACING = [1,0.7,0.7]
NUM_SET = 10
voxelWidthXY = 36
voxelWidthZ  = 24
CAND_THRESHOLD = 150000
outputFile = []
outputFile.append('seriesuid,coordX,coordY,coordZ,probability')

# Keras Parameters
batch_size = 32
num_classes = 2
epochs = 1
input_shape = (1, voxelWidthZ, voxelWidthXY, voxelWidthXY)

K.set_image_data_format('channels_first')

model = Sequential()
model.add(Conv3D(64, kernel_size=(3, 5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.20))
model.add(Conv3D(64, (3, 5, 5), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Save the model in order to restart it later
model.save_weights('model.h5')

writeToDebugFile('Keras Model Created')

candidatesList = Provider.readCSV(candidates)
writeToDebugFile('candidates Read')

for outerSet in range(NUM_SET):
    # Restart the Model
    model.load_weights('model.h5')

    # Create test folder
    subsetDir = originalSubsetDirectoryBase + str(outerSet)
    list = listdir(subsetDir)
    subsetOuterList = []

    # Create Test Set
    for file in list:
        if file.endswith(".mhd"):
            file = file[:-4]
            subsetOuterList.append(file)

    writeToDebugFile('True Positives Starting...')
    # Find True Positives
    truePositiveList = []
    truePositiveLabels = []
    for cand in candidatesList:
        if (cand[4] == '1') and (not (cand[0] in subsetOuterList)):
            fileName = cand[0]+'.mhd'
            for i in range(NUM_SET):
                if (i == outerSet):
                    continue
                else:
                    folderName = originalSubsetDirectoryBase + str(i)
                    fileList = listdir(folderName)
                    if fileName in fileList:
                        originalFilePath = folderName +'/'+cand[0] + '.mhd'
                        newFilePath = resampledSubsetDirectoryBase+str(i) + '/' + cand[0] + '.npy'

                        volumeImage, numpyOrigin, numpySpacing = Provider.load_itk_image(originalFilePath)
                        newVolume = np.load(newFilePath)

                        voxelWorldCoor = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
                        newGeneratedCoor = Provider.worldToVoxelCoord(voxelWorldCoor, numpyOrigin, RESIZE_SPACING)

                        patch = newVolume[int(newGeneratedCoor[0] - voxelWidthZ / 2):int(newGeneratedCoor[0] + voxelWidthZ / 2),
                                int(newGeneratedCoor[1] - voxelWidthXY / 2):int(newGeneratedCoor[1] + voxelWidthXY / 2),
                                int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]

                        try:
                            if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                                zoomFactor = [voxelWidthZ / float(np.shape(patch)[0]),
                                              voxelWidthXY / float(np.shape(patch)[1]),
                                              voxelWidthXY / float(np.shape(patch)[2])]
                                patch = ndimage.zoom(patch, zoom=zoomFactor)
                        except Exception, e:
                            writeToDebugFile('ERROR on TP: ' +str(e)+' -> '+ cand[0])
                            patch = np.zeros((voxelWidthZ, voxelWidthXY, voxelWidthXY))


                        patch = Provider.normalizePlanes(patch)
                        ### ----------- AUGMENTATION ------------
                        ## ----- 0 Degree -----
                        # Original
                        truePositiveList.append(patch)
                        truePositiveLabels.append(1)

                        # 0 flip
                        patchAugmented = np.flip(patch, 0)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)

                        # 1 Flip
                        patchAugmented = np.flip(patch, 1)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)

                        # 2 Flip
                        patchAugmented = np.flip(patch, 2)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)

                        ## ----- 90 Degree -----
                        patch = np.rot90(patch, axes=(1,2))
                        # Original
                        truePositiveList.append(patch)
                        truePositiveLabels.append(1)
                        # 0 flip
                        patchAugmented = np.flip(patch, 0)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)
                        # 1 Flip
                        patchAugmented = np.flip(patch, 1)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)
                        # 2 Flip
                        patchAugmented = np.flip(patch, 2)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)

                        ## ----- 180 Degree -----
                        patch = np.rot90(patch, axes=(1, 2))
                        # Original
                        truePositiveList.append(patch)
                        truePositiveLabels.append(1)
                        # 0 flip
                        patchAugmented = np.flip(patch, 0)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)
                        # 1 Flip
                        patchAugmented = np.flip(patch, 1)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)
                        # 2 Flip
                        patchAugmented = np.flip(patch, 2)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)

                        ## ----- 270 Degree -----
                        patch = np.rot90(patch, axes=(1, 2))
                        # Original
                        truePositiveList.append(patch)
                        truePositiveLabels.append(1)
                        # 0 flip
                        patchAugmented = np.flip(patch, 0)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)
                        # 1 Flip
                        patchAugmented = np.flip(patch, 1)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)
                        # 2 Flip
                        patchAugmented = np.flip(patch, 2)
                        truePositiveList.append(patchAugmented)
                        truePositiveLabels.append(1)


                        break       # if TruePositive candidate found, do not look for other subsets
    writeToDebugFile('True Postives Finished, Total: '+ str(len(truePositiveList)))

    candIndex = 0
    newVolume = np.array([1])
    numpyOrigin = np.array([1])
    while candIndex <= len(candidatesList):
        #Create False Positive List
        falsePositiveList = []
        falsePositiveLables = []
        previousFileName = ''
        while candIndex <= len(candidatesList):
            cand = candidatesList[candIndex]
            candIndex = candIndex + 1
            if (cand[4] == '0') and (not (cand[0] in subsetOuterList)):
                if (previousFileName != cand[0]):
                    previousFileName = cand[0]
                    fileName = cand[0] + '.mhd'
                    for i in range(NUM_SET):
                        if (i == outerSet):
                            continue
                        else:
                            folderName = originalSubsetDirectoryBase + str(i)
                            fileList = listdir(folderName)
                            if fileName in fileList:
                                originalFilePath = folderName + '/' + cand[0] + '.mhd'
                                newFilePath = resampledSubsetDirectoryBase + str(i) + '/' + cand[0] + '.npy'

                                volumeImage, numpyOrigin, numpySpacing = Provider.load_itk_image(originalFilePath)
                                newVolume = np.load(newFilePath)

                                voxelWorldCoor = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
                                newGeneratedCoor = Provider.worldToVoxelCoord(voxelWorldCoor, numpyOrigin,
                                                                              RESIZE_SPACING)

                                patch = newVolume[
                                        int(newGeneratedCoor[0] - voxelWidthZ / 2):int(
                                            newGeneratedCoor[0] + voxelWidthZ / 2),
                                        int(newGeneratedCoor[1] - voxelWidthXY / 2):int(
                                            newGeneratedCoor[1] + voxelWidthXY / 2),
                                        int(newGeneratedCoor[2] - voxelWidthXY / 2):int(
                                            newGeneratedCoor[2] + voxelWidthXY / 2)]

                                try:
                                    if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                                        zoomFactor = [voxelWidthZ / float(np.shape(patch)[0]),
                                                      voxelWidthXY / float(np.shape(patch)[1]),
                                                      voxelWidthXY / float(np.shape(patch)[2])]
                                        patch = ndimage.zoom(patch, zoom=zoomFactor)
                                except Exception, e:
                                    writeToDebugFile('ERROR on FP: ' + str(e)+' -> ' + cand[0])
                                    patch = np.zeros((voxelWidthZ, voxelWidthXY, voxelWidthXY))

                                patch = Provider.normalizePlanes(patch)
                                falsePositiveList.append(patch)
                                falsePositiveLables.append(0)
                                break # if FalsePositive candidate found, do not look for other subsets
                else:
                    voxelWorldCoor = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
                    newGeneratedCoor = Provider.worldToVoxelCoord(voxelWorldCoor, numpyOrigin, RESIZE_SPACING)

                    patch = newVolume[
                            int(newGeneratedCoor[0] - voxelWidthZ / 2):int(newGeneratedCoor[0] + voxelWidthZ / 2),
                            int(newGeneratedCoor[1] - voxelWidthXY / 2):int(newGeneratedCoor[1] + voxelWidthXY / 2),
                            int(newGeneratedCoor[2] - voxelWidthXY / 2):int(newGeneratedCoor[2] + voxelWidthXY / 2)]

                    try:
                        if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                            zoomFactor = [voxelWidthZ / float(np.shape(patch)[0]),
                                          voxelWidthXY / float(np.shape(patch)[1]),
                                          voxelWidthXY / float(np.shape(patch)[2])]
                            patch = ndimage.zoom(patch, zoom=zoomFactor)
                    except Exception, e:
                        writeToDebugFile('ERROR on FP: ' +str(e) +' -> '+cand[0])
                        patch = np.zeros((voxelWidthZ, voxelWidthXY, voxelWidthXY))

                    patch = Provider.normalizePlanes(patch)
                    falsePositiveList.append(patch)
                    falsePositiveLables.append(0)

                if (len(falsePositiveList) == len(truePositiveList)):
                    break

        # If the candidate list is over
        if (len(falsePositiveList) < len(truePositiveList)):
            break

        try:
            totalList = Provider.mixArraysNumpy2(truePositiveList, falsePositiveList, voxelWidthZ, voxelWidthXY)
            totalLabel = Provider.mixArrays(truePositiveLabels, falsePositiveLables)

            # totalList = np.array(totalList)
            totalLabel = np.array(totalLabel)

            totalList = totalList.reshape(totalList.shape[0], 1, voxelWidthZ, voxelWidthXY, voxelWidthXY)
            totalLabel = keras.utils.to_categorical(totalLabel, num_classes)

            writeToDebugFile('Model Train Starting -> ' + str(candIndex))

            hist = model.fit(totalList, totalLabel, batch_size=batch_size, epochs=epochs, verbose=1)
            writeToDebugFile(str(hist.history))
	    	
            writeToDebugFile('Model Train Finished -> ' + str(candIndex))
            writeToDebugFile('Model acc: '+str(hist.history['acc'][0]))

            # if (float(hist.history['acc'][0]) > 0.98):
            #     writeToDebugFile('Early Stopping on Set -> ' + str(outerSet))
            #     break

            if (candIndex>CAND_THRESHOLD):
                writeToDebugFile('Early Stopping on Set -> ' + str(outerSet))
                break

        except Exception, e:
            writeToDebugFile('ERROR on Training ->' + str(e))


    model.save('LUNA16_Model.h5')

    writeToDebugFile('Test Starting -> '+ str(outerSet))
    # *** TEST ***
    for file in subsetOuterList:
        testList = []
        testPatchList = []

        for cand in candidatesList:
            if (cand[0] == file):
                testList.append(cand)
	
        originalFilePath = originalSubsetDirectoryBase+ str(outerSet) + '/' + file + '.mhd'
        newFilePath = resampledSubsetDirectoryBase + str(outerSet) + '/' + file + '.npy'

        volumeImage, numpyOrigin, numpySpacing = Provider.load_itk_image(originalFilePath)
        newVolume = np.load(newFilePath)

        for candidateItem in testList:
            voxelWorldCoor = np.asarray([float(candidateItem[3]), float(candidateItem[2]), float(candidateItem[1])])
            newGeneratedCoor = Provider.worldToVoxelCoord(voxelWorldCoor, numpyOrigin,
                                                          RESIZE_SPACING)

            patch = newVolume[
                    int(newGeneratedCoor[0] - voxelWidthZ / 2):int(
                        newGeneratedCoor[0] + voxelWidthZ / 2),
                    int(newGeneratedCoor[1] - voxelWidthXY / 2):int(
                        newGeneratedCoor[1] + voxelWidthXY / 2),
                    int(newGeneratedCoor[2] - voxelWidthXY / 2):int(
                        newGeneratedCoor[2] + voxelWidthXY / 2)]

            try:
                if np.shape(patch) != (voxelWidthZ, voxelWidthXY, voxelWidthXY):
                    zoomFactor = [voxelWidthZ / float(np.shape(patch)[0]),
                                  voxelWidthXY / float(np.shape(patch)[1]),
                                  voxelWidthXY / float(np.shape(patch)[2])]
                    patch = ndimage.zoom(patch, zoom=zoomFactor)
            except Exception, e:
                writeToDebugFile('ERROR on Test List: ' +str(e)+' -> '+ candidateItem[0])
                patch = np.zeros((voxelWidthZ, voxelWidthXY, voxelWidthXY))

            patch = Provider.normalizePlanes(patch)
            testPatchList.append(patch)

        try:
            testPatchList = np.asarray(testPatchList)
            testPatchList = testPatchList.reshape(testPatchList.shape[0], 1 ,voxelWidthZ, voxelWidthXY, voxelWidthXY)

            predictions = model.predict(testPatchList, batch_size=32, verbose=0)
        except Exception, e:
            writeToDebugFile('ERROR on Test Predict: -> ' + str(e))

        writeToDebugFile('Recording Predictions...')
	
        try:
            for i in range(len(testList)):
                line = testList[i][0] +','+ testList[i][1]+','+testList[i][2]+','+testList[i][3]+','+str(predictions[i][1])
                outputFile.append(line)
        except Exception, e:
            writeToDebugFile('ERROR on testList append -> '+ str(e))

    #with open('test_'+str(outerSet)+'.csv','w') as myFile:
    #    for item in outputFile:
    #        myFile.write('%s\n' % item)

    writeToDebugFile('SUBSET -> '+str(outerSet)+' is OVER')

#Write to File
with open('METU_VISION_FPRED.csv','w') as myFile:
    for item in outputFile:
        myFile.write('%s\n' % item)
