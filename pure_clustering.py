import numpy as np
import sklearn
from keras.datasets import mnist
from sklearn.cluster import KMeans

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x = np.concatenate((x_train, x_test))
#y = np.concatenate((y_train, y_test))
x_train = x_train.reshape((x_train.shape[0], -1))
x_train = np.divide(x_train, 255.)
x_test = x_test.reshape((x_test.shape[0], -1))
x_test = np.divide(x_test, 255.)
n_clusters = len(np.unique(y_train))
print("n_clusters = ",n_clusters)
print("x_train shae: ",x_train.shape)
print("x_test shape: ",x_test.shape)

kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(x_train)
y_pred_kmeans = kmeans.predict(x_test)

print(sklearn.metrics.rand_score(y_test, y_pred_kmeans))