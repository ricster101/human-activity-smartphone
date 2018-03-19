import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import mstats
from numpy import genfromtxt
import scipy.io

datasetDir = 'dataset/'
trainDir = 'train/'
inert = 'Inertial Signals/'
dataP = 'X_train.txt'
labelsP = 'y_train.txt'

data = open(datasetDir+trainDir+dataP, 'r')
labels = open(datasetDir+trainDir+labelsP)

dataNp = np.loadtxt(data)
labelsNp = np.loadtxt(labels)
labelsNp = np.array(labelsNp)
#0 - Stopped
# 1 - Walking

labelsNpBin = np.zeros(shape=labelsNp.shape)
for i,j in enumerate(labelsNp):
    if (j == 1 or j == 2 or j ==3):
        labelsNpBin[i] = 1
    else:
        labelsNpBin[i] = 0


def lda2(dataNp , labelsNp, labelsNum ):

    labelsNpBin = np.array(labelsNp)
    ########### LDA step by step ##############

    # 1. Mean for each class

    mean_vectors =[]
    for cl in range(0,labelsNum):
        mean_vectors.append(np.mean(dataNp[labelsNpBin==cl],axis=0))
        #print('Mean Vector class %s: %s\n' % (cl, mean_vectors[cl - 1]))

    mean_vectors = np.array(mean_vectors)
    #print(mean_vectors.shape)

    # 2. Computing scater matrices

    S_W = np.zeros((561,561))
    for cl,mv in zip(range(0,labelsNum), mean_vectors):
        class_sc_mat = np.zeros((561,561))                  # scatter matrix for every class
        for row in dataNp[labelsNpBin == cl]:
            row, mv = row.reshape(561,1), mv.reshape(561,1) # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat                             # sum class scatter matrices
    #print('within-class Scatter Matrix:\n', S_W)

    overall_mean = np.mean(dataNp, axis=0)

    S_B = np.zeros((561,561))
    for i,mean_vec in enumerate(mean_vectors):
        n = dataNp[labelsNpBin==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(561,1) # make column vector
        overall_mean = overall_mean.reshape(561,1) # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    #print('between-class Scatter Matrix:\n', S_B)
    #print(S_B.shape)

    # 3. Solving generalized Eigenvalues Problem
    # Nao e possivle calcular a inversa pois temos uma matriz singular
    eig_vals, eig_vecs = np.linalg.eig(S_W.dot(S_B))

    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(561,1)
        #print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
        #print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

    # 4. Selecting Linear discriminants

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    W = np.hstack((eig_pairs[0][1].reshape(561,1), eig_pairs[1][1].reshape(561,1)))

    #print(W.real)

    X_lda = dataNp.dot(W)

    y=labelsNpBin
    def plot_step_lda():

        ax = plt.subplot(111)
        for label,marker,color in zip(
                range(0,labelsNum),( 'o','o'),( 'red', 'green')):

            plt.scatter(x=X_lda[:,0].real[y == label],
                        y=X_lda[:,1].real[y == label],
                        marker=marker,
                        color=color,
                        alpha=0.5,
                        label=label
                        )

        plt.xlabel('LD1')
        plt.ylabel('LD2')

        leg = plt.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.title('LDA - 2 classes')

        # hide axis ticks
        plt.tick_params(axis="both", which="both", bottom="off", top="off",
                        labelbottom="on", left="off", right="off", labelleft="on")

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.grid()
        plt.tight_layout
        plt.show()

    plot_step_lda()
    #np.savetxt('lda_binary.txt',X_lda,delimiter = ';' )
    #np.savetxt('lda_binary_labels.txt',y,delimiter = ';')

lda2(dataNp,labelsNpBin,2)

# LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(dataNp, labelsNp)

lda2dbinary = LDA(n_components=2)
X_lda_2dbinary = lda2dbinary.fit_transform(dataNp,labelsNpBin)

#np.savetxt('lda_binary.txt',X_lda_2dbinary,delimiter = ';' )
#np.savetxt('lda_binary_labels.txt',labelsNpBin,delimiter = ';')

def plot_scikit_lda(X, y,title):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(0,6),('o', 'o', 'o', 'o', 'o', 'o'),('blue', 'red', 'green','purple','yellow','black')):

        plt.scatter(x=X[:,0][y == label],
                    y=X[:,1][y == label] * -1, # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label
                     )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()


plot_scikit_lda(X_lda_sklearn,labelsNp,"LDA - 6 classes")
#np.savetxt('lda_6classes.txt',X_lda_sklearn)
#np.savetxt('lda6classes_labels.txt',labelsNp)