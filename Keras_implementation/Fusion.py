from os import listdir
import Provider

# Directories
ModelDirectory = r'ModelResults'

ModelList = []
outputFile = []
outputFile.append('seriesuid,coordX,coordY,coordZ,probability')

list = listdir(ModelDirectory)

for modelFile in list:
    path = ModelDirectory+'/'+modelFile
    ModelList.append(Provider.readCSV(path))

NumberOfCandidates = len(ModelList[0])
NumberOfModel = len(ModelList)

for i in range(NumberOfCandidates):
    if (i == 0) :
        continue
    average = 0
    for j in range(NumberOfModel):
        average = average + float(ModelList[j][i][4])

    average = average / NumberOfModel

    outputFile.append(ModelList[0][i][0]+','+ModelList[0][i][1]+','+ModelList[0][i][2]+','+ModelList[0][i][3]+','+str(average))

    print (str(i)+'\n')

#Write to File
with open('METU_VISION_FPRED.csv','w') as myFile:
    for item in outputFile:
        myFile.write('%s\n' % item)
