import sklearn.metrics as sk
import numpy as np
import kruskal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import itertools
import kruskal

datasetDir = 'dataset/'
trainDir = 'train/'
inert = 'Inertial Signals/'
dataP = 'X_train.txt'
labelsP = 'y_train.txt'
testDir = 'test/'
dataTestX = 'X_test.txt'
dataTesty = 'y_test.txt'

#data = open(datasetDir+trainDir+dataP, 'r')
labels = open(datasetDir+trainDir+labelsP)


dataNp = np.loadtxt(datasetDir+trainDir+dataP)
labelsNp = np.loadtxt(datasetDir+trainDir+labelsP)
labelsNp = np.array(labelsNp)

testDataX = np.loadtxt(datasetDir+testDir+dataTestX)
testDatay = np.loadtxt(datasetDir+testDir+dataTesty)

#print(testDataX)
labelsNpBin = np.zeros(shape=labelsNp.shape)
for i,j in enumerate(labelsNp):
    if (j == 1 or j == 2 or j ==3):
        labelsNpBin[i] = 1
    else:
        labelsNpBin[i] = 0

LDAdata = np.loadtxt('lda_binary.txt',delimiter=';')
LDAdataLabels = np.loadtxt('lda_binary_labels.txt',delimiter = ';')

bestFeatures = kruskal.sendVars()

#confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)[source]


# Base data set
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(dataNp,labelsNpBin)
#testDataX = sklearn_lda.transform(testDataX)
y_pred = sklearn_lda.predict(testDataX)

labelsNpBin2 = np.zeros(shape=testDatay.shape)
for i,j in enumerate(testDatay):
    if (j == 1 or j == 2 or j ==3):
        labelsNpBin2[i] = 1
    else:
        labelsNpBin2[i] = 0

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



#plot_confusion_matrix(sk.confusion_matrix(labelsNpBin2,y_pred),['0 1'])


kruskalVars = kruskal.sendVars()
newData = []
for i in kruskalVars:
    newData.append(dataNp[i])

sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(newData,labelsNpBin)
#testDataX = sklearn_lda.transform(testDataX)
y_pred = sklearn_lda.predict(testDataX)

plot_confusion_matrix(sk.confusion_matrix(labelsNpBin2,y_pred),['0 1'])