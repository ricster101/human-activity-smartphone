import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
import sklearn.discriminant_analysis as lda
import kruskal
import minDistClass
from sklearn.decomposition import PCA as skPca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#multiplicar o Wt pelas features de teste


datasetDir = 'dataset/'
trainDir = 'train/'
dataP = 'X_train.txt'
labelsP = 'y_train.txt'

data = open(datasetDir+trainDir+dataP, 'r')
labels = open(datasetDir+trainDir+labelsP)


#test Set
testDir = 'test/'
dataT = 'X_test.txt'
labelsT = 'y_test.txt'

dataTest = open(datasetDir+testDir+dataT,'r')
labelsTest = open(datasetDir+testDir+labelsT, 'r')


dataNp = np.loadtxt(data)
labelsNp = np.loadtxt(labels)

testData = np.loadtxt(dataTest)
testLabels = np.loadtxt(labelsTest)


labelsNpBin = np.zeros(shape=labelsNp.shape)
for i,j in enumerate(labelsNp):
    if (j == 1 or j == 2 or j ==3):
        labelsNpBin[i] = 1
    else:
        labelsNpBin[i] = 0

testlabelsBin = np.zeros(shape=testLabels.shape)
for i,j in enumerate(testLabels):
    if (j == 1 or j == 2 or j ==3):
        testlabelsBin[i] = 1
    else:
        testlabelsBin[i] = 0

kruskalVars = kruskal.sendVars()

minDistClass.classifier(dataNp, labelsNpBin, testData,testlabelsBin, True)

dataKruskal = dataNp[:, [kruskalVars.astype(int)]]
dataKruskal = dataKruskal.squeeze()

testdataKruskal = testData[:, [kruskalVars.astype(int)]]
testdataKruskal = testdataKruskal.squeeze()

minDistClass.classifier(dataKruskal, labelsNpBin, testdataKruskal,testlabelsBin, True)

dadosPca = skPca(n_components=20)
dataPosPca = dadosPca.fit_transform(dataNp)
testPosPca = dadosPca.transform(testData)

minDistClass.classifier(dataPosPca, labelsNpBin, testPosPca, testlabelsBin, True)

sklearn_lda = LDA(n_components=3)
dataPosLda = sklearn_lda.fit_transform(dataNp,labelsNpBin)
testPosLda = sklearn_lda.transform(testData)

minDistClass.classifier(dataPosLda, labelsNpBin, testPosLda, testlabelsBin, True)

#dataNpCentered = dataNp - dataNp.mean(axis=1).reshape(-1, 1)

'''
results = PCA(dataNp)

x = []
y = []
z = []
for item in results.Y:
    x.append(item[0])
    y.append(item[1])
    z.append(item[2])


plt.figure()
plt.plot(results.s[0:50])
plt.show()


#plt.close('all')  # close all latent plotting windows
fig1 = plt.figure()  # Make a plotting figure
ax = Axes3D(fig1)  # use the plotting figure to create a Axis3D object.
pltData = [x, y, z]
n=labelsNp.astype(int)
colors = [int(i % 23) for i in n]
ax.scatter(pltData[0], pltData[1], pltData[2], c=colors)  # make a scatter plot of blue dots from the data

# make simple, bare axis lines through space:
xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0, 0))  # 2 points make the x-axis line at the data extrema along x-axis
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')  # make a red line for the x-axis.
yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0, 0))  # 2 points make the y-axis line at the data extrema along y-axis
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')  # make a red line for the y-axis.
zAxisLine = ((0, 0), (0, 0), (min(pltData[2]), max(pltData[2])))  # 2 points make the z-axis line at the data extrema along z-axis
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')  # make a red line for the z-axis.

# label the axes
ax.set_xlabel("x-axis label")
ax.set_ylabel("y-axis label")
ax.set_zlabel("z-axis label")
ax.set_title("The title of the plot")
'''