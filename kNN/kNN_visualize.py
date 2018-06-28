import matplotlib
import matplotlib.pyplot as plt
import numpy
from numpy import array

def drawDatingEx(datingDataMat, datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels, dtype=numpy.int32), 15.0*array(datingLabels, dtype=numpy.int32))
    plt.show()
