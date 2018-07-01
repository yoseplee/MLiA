# -*- coding: utf-8 -*-

# this is for implementing decision tree algorithm
# from machine learning in action
# 2018.06.30

from math import log
import operator

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
    baseEntropy = calcShannonEnt(dataSet)   #calculate entropy based on yes or no
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] #get all rows for i-th column
        uniqueVals = set(featList)  # remove all duplicates
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet)) # for this variable prob, it means ratio: n(subDataSet) / nAll
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]