from os import listdir
import numpy as np
import Provider
import tensorflow as tf

# Directories
subsetDirBase = r'C:\Users\polat\Desktop\TEZ\LUNA\subset'
candidates = r'C:\Users\polat\Desktop\TEZ\LUNA\CSVFILES\candidates_V2.csv'

# Constants
NUM_SUBSET = 10
voxelWidth = 42
ZWidth = 2

candidatesList = Provider.readCSV(candidates)

# ***** TRAINING STARTS*****
inputVolume = tf.placeholder(tf.float32)
output = tf.placeholder(tf.float32)

W_conv1 = Provider.weight_variable([3, 3, 3, 1, 32])
B_conv1 = Provider.bias_variable([32])

h_conv1 = tf.nn.relu(tf.nn.conv3d(inputVolume, W_conv1, [1, 1, 1, 1, 1], padding='SAME') + B_conv1)
layer1 = tf.nn.max_pool3d(h_conv1, ksize=[1, 2, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

W_conv2 = Provider.weight_variable([3, 3, 3, 32, 64])
B_conv2 = Provider.bias_variable([64])

h_conv2 = tf.nn.relu(tf.nn.conv3d(layer1, W_conv2, [1, 1, 1, 1, 1], padding='SAME') + B_conv2)
layer2 = tf.nn.max_pool3d(h_conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

# FullyConnected1 = tf.reshape(layer2, [-1, np.shape(layer2)[0] * np.shape(layer2)[1] * np.shape(layer2)[2]])
FullyConnected1 = tf.reshape(layer2, [-1, 7 * 7 * 3])

# W_FullyConnected = Provider.weight_variable(
#     [np.shape(layer2)[0] * np.shape(layer2)[1] * np.shape(layer2)[2], 1024])
W_FullyConnected = Provider.weight_variable(
    [7 * 7 * 3, 1024])
B_FullyConnected1 = Provider.bias_variable([1024])

hiddenL = tf.nn.relu(tf.matmul(FullyConnected1, W_FullyConnected) + B_FullyConnected1)

W_FullyConnected2 = Provider.weight_variable([1024, 2])
B_FullyConnected2 = Provider.bias_variable([2])

prediction = tf.nn.softmax(tf.matmul(hiddenL, W_FullyConnected2) + B_FullyConnected2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, output))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
tf.summary.scalar("Cross Entropy", cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy", accuracy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Summary Writer
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('/Logs', sess.graph)
# ***** TRAINING ENDS *****


for outerSet in range(NUM_SUBSET):
    # Create test folder
    subsetDir = subsetDirBase + str(outerSet)
    list = listdir(subsetDir)
    subsetList = []

    # Create Test Set
    for file in list:
        if file.endswith(".mhd"):
            file = file[:-4]
            subsetList.append(file)

    # Find True Positives
    truePositiveList = []
    for cand in candidatesList:
        if (cand[4] == '1') and (not (cand[0] in subsetList)):
            truePositiveList.append(cand)

    candIndex = 0
    while candIndex <= len(candidatesList):
        # Create False Positive List
        falsePositiveList = []
        while candIndex <= len(candidatesList):
            cand = candidatesList[candIndex, :]
            if (cand[4] == '0') and (not (cand[0] in subsetList)):
                falsePositiveList.append(cand)
            cand = cand + 1
            if (len(falsePositiveList) == len(truePositiveList)):
                break
        # Eğer candidatesList bitmiş ise, Candidates listesinin sonuna gelinmiş ise
        if (len(falsePositiveList) < len(truePositiveList)):
            break

        # Mix True Positives and False Negatives
        mixedArray = Provider.mixArrays(truePositiveList, falsePositiveList)

        for step in range(len(mixedArray)):
            fileName = mixedArray[step, 0] + '.mhd'

            for i in range(NUM_SUBSET):
                if (outerSet == i):
                    continue
                folder = subsetDirBase + str(i)
                fileList = listdir(folder)
                if fileName in fileList:
                    filePath = folder + '\\' + fileName
                    volumeImage, numpyOrigin, numpySpacing = Provider.load_itk_image(filePath)

                    voxelWorldCoor = np.asarray(
                        [float(mixedArray[step, 3]), float(mixedArray[step, 2]), float(mixedArray[step, 1])])
                    voxelPixelCoord = Provider.worldToVoxelCoord(voxelWorldCoor, numpyOrigin, numpySpacing)
                    patch = volumeImage[voxelPixelCoord[0] - ZWidth / 2:voxelPixelCoord[0] + ZWidth / 2,
                            voxelPixelCoord[1] - voxelWidth / 2:voxelPixelCoord[1] + voxelWidth / 2,
                            voxelPixelCoord[2] - voxelWidth / 2:voxelPixelCoord[2] + voxelWidth / 2]

                    patch = Provider.normalizePlanes(patch)
                    result = Provider.makeOutputArray(mixedArray[step, 4])
                    sess.run(train_step, feed_dict={inputVolume: patch, output: mixedArray[step, 4]})
                    if ((step % 20) == 0):
                        summary = sess.run(merged, feed_dict={inputVolume: patch, output: result })
                        writer.add_summary(summary)