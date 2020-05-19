import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D


features = np.load('../../../../exp/pcnn/features_pcnn2.npy')
labels = np.load('../../../../exp/pcnn/labels_pcnn2.npy')

lda = LinearDiscriminantAnalysis(n_components=3)
pca = PCA(n_components=3)

#data = lda.fit_transform(features, labels)
data = pca.fit_transform(features, labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax0 = ax.scatter(data[:,0], data[:,1], data[:,2], c=labels, cmap='rainbow')

fig.colorbar(ax0)
plt.show()

