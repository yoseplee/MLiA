#2018.07.19 naive bayes

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worhless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 is abusive, 0 is not
    return postingList, classVec

def createVocabList(dataSet):
    #create a list of all the unique words in all of our documents
    vocabSet = set([])              #create empty set
    for document in dataSet:
        print("create the union of two sets:: " + document)
        vocabSet = vocabSet | set(document)     # create the union of two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    #takes vocab list and a document and outputs a vector of 1s and 0s to represent
    #whether a word from our vocab is present or not in the given document
    returnVec = [0]*len(vocabList)      #create a vector of all 0s
    for word in inputSet:
        if word in vocabList:           #if there is that word in vocablist, than flag it
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec