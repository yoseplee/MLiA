# -*- coding: utf-8 -*-
# 2018-06-29
# Machine Learning in action / kNN
# page ~33

from numpy import *
import operator
from os import listdir  # to see the names of files in a given directory

def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # inX: input vector, dataSet: training data, labels: target variable, k: nearest neighbors
    dataSetSize = dataSet.shape[0] # shape returns (nRows/nCols)
    diffMat = tile(inX, (dataSetSize,1)) - dataSet # tile: inX의 값으로 (dataSetSize,1) 행렬 만들기
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) # axis=1 행, =0 열
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() # index만 소팅함
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 파일을 파싱하여 kNN 분류기에서 사용 가능한 형태로 가공한다.
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    numberOfCols = getnTab(filename)
    returnMat = zeros((numberOfLines, numberOfCols)) #조금더 adaptive하게 만들 수 있다. -> getTam(filename)
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1])) # in case using(coded) 'datingTestSet2.txt'
        # classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector

# normalization -> 모든 값을 0~1사이로 변환하여 각 값이 가지는 가중 요소를 정제하는 것
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0] # nCols of the dataSet
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1)) # element-wise division
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #classify0(data that to classify, test data set, labels, n neighbors)
        classfierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %s, the real answer is: %s" % (classfierResult, datingLabels[i]))
        if(classfierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percenstTats = float(raw_input("Percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percenstTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", resultList[classifierResult - 1]

# For example of handwritting recognition system
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()                     #000000000111100000... for 32 digits
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])  #because returnVec has single rows, its idx is 0.
    return returnVect

def handwrittingClassTest():
    dirName = 'digits/testDigits'
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    testFileList = listdir(dirName)
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('%s/%s' % (dirName, fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if(classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of error is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

# *********additional functions********* #


# from data file(text) seperated by \t, count the number of tabs in a single line.
def getnTab(filename):
    count = 0
    fr = open(filename)
    line = fr.readline()
    for ch in line:
        if(ch == '\t'):
            count += 1
    return count

