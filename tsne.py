from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn import datasets
import numpy as np
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import data_load


path = 'feature/MCF7/train/RR/'
first = 'left'
second = 'right'
y = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
feature1_1 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,1:]
feature1_2 = np.loadtxt(path + 'RCKmer_'+second+'.txt')[:,1:]
feature2_1 = np.loadtxt(path + 'PCSF_'+first+'.txt')
feature2_2 = np.loadtxt(path + 'PCSF_'+second+'.txt')
x3_1 =np.loadtxt(path + 'PSTNPds_'+first+'.txt')[:,1:]
x3_2 =np.loadtxt(path + 'PSTNPds_'+second+'.txt')[:,1:]
x4_1 =np.loadtxt(path + 'PSTNPss_'+first+'.txt')[:,1:]
x4_2 =np.loadtxt(path + 'PSTNPss_'+second+'.txt')[:,1:]
feature5_1 = np.loadtxt(path + 'NPSE_'+first+'.txt')
feature5_2 = np.loadtxt(path + 'NPSE_'+second+'.txt')
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2,feature5_1,feature5_2, y)



# x_train,x_test, y_train, y_test =train_test_split(data,y,test_size=0.2, random_state=10)
tsne = TSNE(n_components=2, init='pca', perplexity=100,learning_rate=300,random_state=10)
x = tsne.fit_transform(x)
y_pred = KMeans(n_clusters = 2,random_state=10).fit_predict(x)
plt.figure()
colors = ['navy', 'red']
target_names = ['negative','positive']
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1],  color=color,
                s=1,label=target_name)
x_max = np.max(x[:,0]) + 1
x_min = np.min(x[:,0]) - 1
y_max = np.max(x[:,1]) + 1
y_min = np.min(x[:,1]) - 1
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.show()
