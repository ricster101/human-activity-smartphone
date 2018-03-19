import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
import sklearn.discriminant_analysis as lda
from scipy.stats import mstats
from numpy import genfromtxt
import scipy.io

kruskalChiSq = scipy.io.loadmat('chiSq.mat')

kruskalChiSq = kruskalChiSq['chiSq']

kruskalChiSq = kruskalChiSq[kruskalChiSq[:,1].argsort()]

kruskalChiSq = kruskalChiSq[::-1]

contributeMore20 = kruskalChiSq[0:20,0] #indice esta a comecar em 1

def sendVars():
    plt.plot(kruskalChiSq[0:100,1])
    plt.show()
    return contributeMore20



#Acho que nao vou usar o kruskal wallis do python pq nao me da o chi qudarado
#fazer esta parte no matlab para inferir as melhores variaveis
#importtar a matriz do matlab

