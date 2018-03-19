import numpy as np
import scipy.spatial.distance as distanceCalc
import visualConfusionMatrix

def classifier(xtrain, ytrain, xtest, ytest, binary):
    indexMax = (xtrain.shape[1])
    print(xtest.shape)
    predict = []
    if binary:
        class0 = xtrain[np.where(ytrain[:] == 0)]
        class1 = xtrain[np.where(ytrain[:] == 1)]

        meanClass0 = np.mean(class0, axis=0)
        meanClass1 = np.mean(class1, axis=0)

        for point in xtest:
            pointToClass0 = distanceCalc.euclidean(point, meanClass0)
            pointToClass1 = distanceCalc.euclidean(point, meanClass1)

            if pointToClass0<pointToClass1:
                pointLabel = 0
            else:
                pointLabel = 1
            predict.append(pointLabel)

        visualConfusionMatrix.visualConfusionMatrix(ytest,predict, ['0 1'])