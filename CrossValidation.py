from os import listdir
import numpy as np
import Provider

# Directories
subsetDirBase = r'C:\Users\polat\Desktop\TEZ\LUNA\subset'
candidates = r'C:\Users\polat\Desktop\TEZ\LUNA\CSVFILES\candidates_V2.csv'


#Constants
SUBSET = 10

candidatesList = Provider.readCSV(candidates)

for i in range(SUBSET):
    # Create test folder
    subsetDir = subsetDirBase + str(i)
    list = listdir(subsetDir)
    subsetList = []

    # Create Test Set
    for file in list:
        if file.endswith(".mhd"):
            file = file[:-4]
            subsetList.append(file)

    #Find True Positives
    truePositiveList = []
    for cand in candidatesList:
        if (cand[4]==1) and (not (cand[0] in subsetList)):
            truePositiveList.append(cand)

    candIndex = 0
    while candIndex <=  len(candidatesList):
        # Create False Positive List
        falsePositiveList = []
        while candIndex <= len(candidatesList):
            cand = candidatesList[candIndex,:]
            if (cand[4] == 0) and (not (cand[0] in subsetList)):
                falsePositiveList.append(cand)
            cand = cand + 1
            if (len(falsePositiveList)==len(truePositiveList)):
                break

        if(len(falsePositiveList)<len(truePositiveList))
            continue

        #Mix True Positives and False Negatives
        mixedArray = mixArrays(truePositiveList, falsePositiveList)

