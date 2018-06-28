# -*- coding: utf-8 -*-
from numpy import *
import operator

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
    returnMat = zeros((numberOfLines, numberOfCols)) #조금더 adaptive하게 만들 수 있다. 방법은?
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # classLabelVector.append(int(listFromLine[-1])) # in case using(coded) 'datingTestSet2.txt'
        classLabelVector.append(listFromLine[-1])
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

