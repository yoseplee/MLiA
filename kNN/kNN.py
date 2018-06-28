from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    #inX: input vector, dataSet: training data, labels: target variable, k: nearest neighbors
    dataSetSize = dataSet.shape[0] #shape returns (행,열)
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #tile: inX의 값으로 (dataSetSize,1) 행렬 만들기
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #axis=1 행, =0 열
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort() #index만 소팅함
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3)) #numpy zeros
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
