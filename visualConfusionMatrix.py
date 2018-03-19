import plot_confusion_matrix as pltCM
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def visualConfusionMatrix(groundTruth, preditctions, classNames):
        cMatrix = confusion_matrix(groundTruth, preditctions)

        plt.figure()
        pltCM.plot_confusion_matrix(cMatrix, classNames)

        #plt.show()

