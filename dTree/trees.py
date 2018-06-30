# -*- coding: utf-8 -*-

# this is for implementing decision tree algorithm
# from machine learning in action
# 2018.06.30

from math import log

def calcShannonEnt(dataSet):    #dataSet type is python list
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]              # yes or no, from createDataSet()
        # count on the number of yes and no
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0       #!! a defect in textbook. this line should belong to if condition statement.
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'], [1,1,'yes'], [1,0,'no'], [0,1,'no'], [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]             #[], :axis means that from start to just before axis
            reducedFeatVec.extend(featVec[axis+1:])     #1, yes #axis+1: means that from axis+1 to end
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet): #dataset type is python list
    numFeatures = len(dataSet[0]) -1    #because index starts from 0, but len() returns the number of cols
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
