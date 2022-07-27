import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay

np.random.seed(10)

import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.cluster import KMeans
import sklearn.metrics
from scipy.ndimage import gaussian_filter

"""Pseudocode. 
def x_extremeties
	returns file with dimensions (10000, 4, 1). [x_min, x_left_center, x_right_center, x_max] where 0 <= x_i <= 28
def cluster
	returns cluster score from KMeans and plots confusion matrix

def main
	load data with eventual filter and call every function"""


def x_values(training_data, label_data):
	outer_array = []
	labels = []
	xmin = []
	xmax = []
	picture_index = 0
	while picture_index <= (training_data.shape[0] - 1):
		n_nonzero_pixels = 0
		sum = 0
		for y_index in range(training_data.shape[1]):
			for x_index in range(training_data.shape[2]):
				sum += training_data[picture_index][y_index][x_index]
		if sum <= 0:
			picture_index += 1
		else:
			nonzero_x_array = []
			inner_array = []  # [x_min, x_center_left, x_center_right, x_max, n_nonzero pixels]
			for y_index in range(training_data.shape[1]):
				for x_index in range(training_data.shape[2]):
					if training_data[picture_index][y_index][x_index] > 0:
						#print(picture_index)
						nonzero_x_array.append(x_index)
						n_nonzero_pixels += 1

			left_of_center = [-5]
			right_of_center = [34]
			for elem in nonzero_x_array:
				if elem <= 13:
					left_of_center.append(elem)
				else:
					right_of_center.append(elem)

			inner_array.append(min(nonzero_x_array))
			inner_array.append(max(left_of_center))
			inner_array.append(min(right_of_center))
			inner_array.append(max(nonzero_x_array))
			inner_array.append(n_nonzero_pixels)

			xmin.append(min(nonzero_x_array))
			xmax.append(max(nonzero_x_array))
			# print(inner_array)

			outer_array.append(inner_array)
			labels.append(label_data[picture_index])
			picture_index += 1

	result_data = np.array(outer_array)
	result_label = np.array(labels)

	plt.scatter(xmin, xmax)
	plt.title('Min-Max')
	plt.ylabel('xmax')
	plt.xlabel('xmin')
	plt.show(block=False)

	return [result_data, result_label]

#def file_merge(training_files, testing_files):



def cluster(classes_used, train_data, test_data, test_labels):
	kmeans = KMeans(n_clusters=classes_used, n_init=40).fit(train_data)
	y_pred_kmeans = kmeans.predict(test_data)
	y_pred_kmeans = y_pred_kmeans + 1
	# Scoring
	score = sklearn.metrics.rand_score(test_labels, y_pred_kmeans)
	print("score is: ", score)
	# Confusion matrix
	from sklearn.metrics import confusion_matrix

	cm = confusion_matrix(y_true=test_labels, y_pred=y_pred_kmeans)

	fig = plt.figure()
	import seaborn as sns;
	sns.set()
	ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", )
	plt.title('Confusion matrix')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show(block=False)

	from scipy.optimize import linear_sum_assignment as linear_assignment

	def _make_cost_m(cm):
		s = np.max(cm)
		return (- cm + s)

	indexes = linear_assignment(_make_cost_m(cm))
	js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
	cm2 = cm[:, js]
	# sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
	acc = np.trace(cm2) / np.sum(cm2)
	print("accuracy is: ", acc)
	return y_pred_kmeans

def extract_data(X_test, Y_test, y_pred_kmeans):
	x_11 = []
	x_12 = []
	x_13 = []
	x_14 = []
	x_21 = []
	x_22 = []
	x_23 = []
	x_24 = []
	x_31 = []
	x_32 = []
	x_33 = []
	x_34 = []
	x_41 = []
	x_42 = []
	x_43 = []
	x_44 = []

	for true_index in range(len(Y_test)):

		pred_index = true_index

		if Y_test[true_index] == 1:

			if y_pred_kmeans[pred_index] == 1:
				x_11.append(X_test[true_index])

			elif y_pred_kmeans[pred_index] == 2:
				x_12.append(X_test[true_index])

			elif y_pred_kmeans[pred_index] == 3:
				x_13.append(X_test[true_index])

			else:
				x_14.append(X_test[true_index])

		elif Y_test[true_index] == 2:

			if y_pred_kmeans[pred_index] == 1:
				x_21.append(X_test[true_index])

			elif y_pred_kmeans[pred_index] == 2:
				x_22.append(X_test[true_index])

			elif y_pred_kmeans[pred_index] == 3:
				x_23.append(X_test[true_index])

			else:
				x_24.append(X_test[true_index])

		elif Y_test[true_index] == 3:

			if y_pred_kmeans[pred_index] == 1:
				x_31.append(X_test[true_index])

			elif y_pred_kmeans[pred_index] == 2:
				x_32.append(X_test[true_index])

			elif y_pred_kmeans[pred_index] == 3:
				x_33.append(X_test[true_index])

			else:
				x_34.append(X_test[true_index])

		else:

			if y_pred_kmeans[pred_index] == 1:
				x_41.append(X_test[true_index])

			elif y_pred_kmeans[pred_index] == 2:
				x_42.append(X_test[true_index])

			elif y_pred_kmeans[pred_index] == 3:
				x_43.append(X_test[true_index])

			else:
				x_44.append(X_test[true_index])

	# Plot the pictures

	#x11
	plt.figure(figsize=(80, 4))

	for i in range(len(x_11)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_11[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 1, Predicted label = 1")
	plt.show(block=False)

	#x12
	plt.figure(figsize=(80, 4))

	for i in range(len(x_12)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_12[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 1, Predicted label = 2")
	plt.show(block=False)

	#x13
	plt.figure(figsize=(80, 4))

	for i in range(len(x_13)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_13[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 1, Predicted label = 3")
	plt.show(block=False)

	# x14
	plt.figure(figsize=(80, 4))

	for i in range(len(x_14)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_14[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 1, Predicted label = 4")
	plt.show(block=False)

	# x21
	plt.figure(figsize=(80, 4))

	for i in range(len(x_21)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_21[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 2, Predicted label = 1")
	plt.show(block=False)

	# x22
	plt.figure(figsize=(80, 4))

	for i in range(len(x_22)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_22[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 2, Predicted label = 2")
	plt.show(block=False)

	# x23
	plt.figure(figsize=(80, 4))

	for i in range(len(x_23)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_23[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 2, Predicted label = 3")
	plt.show(block=False)

	# x24
	plt.figure(figsize=(80, 4))

	for i in range(len(x_24)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_24[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 2, Predicted label = 4")
	plt.show(block=False)

	# x31
	plt.figure(figsize=(80, 4))

	for i in range(len(x_31)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_31[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 3, Predicted label = 1")
	plt.show(block=False)

	# x32
	plt.figure(figsize=(80, 4))

	for i in range(len(x_32)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_32[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 3, Predicted label = 2")
	plt.show(block=False)

	# x33
	plt.figure(figsize=(80, 4))

	for i in range(len(x_33)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_33[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 3, Predicted label = 3")
	plt.show(block=False)

	# x34
	plt.figure(figsize=(80, 4))

	for i in range(len(x_34)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_34[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 3, Predicted label = 4")
	plt.show(block=False)

	# x41
	plt.figure(figsize=(80, 4))

	for i in range(len(x_41)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_41[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 4, Predicted label = 1")
	plt.show(block=False)

	# x42
	plt.figure(figsize=(80, 4))

	for i in range(len(x_42)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_42[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 4, Predicted label = 2")
	plt.show(block=False)

	# x43
	plt.figure(figsize=(80, 4))

	for i in range(len(x_43)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_43[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 4, Predicted label = 3")
	plt.show(block=False)

	# x44
	plt.figure(figsize=(80, 4))

	for i in range(len(x_44)):
		if i < 10:
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(x_44[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	plt.title("True label = 4, Predicted label = 4")
	plt.show(block=False)


def main():
	X_test_DD = np.load('current_phys_data/data_DD/test_X.npy', mmap_mode='r')
	X_train_DD = np.load('current_phys_data/data_DD/train_X.npy', mmap_mode='r')
	Y_test_DD = np.load('current_phys_data/data_DD/test_Y.npy', mmap_mode='r')
	Y_train_DD = np.load('current_phys_data/data_DD/train_Y.npy', mmap_mode='r')

	X_test_ND = np.load('current_phys_data/data_ND/test_X.npy', mmap_mode='r')
	X_train_ND = np.load('current_phys_data/data_ND/train_X.npy', mmap_mode='r')
	Y_test_ND = np.load('current_phys_data/data_ND/test_Y.npy', mmap_mode='r')
	Y_train_ND = np.load('current_phys_data/data_ND/train_Y.npy', mmap_mode='r')

	X_test_SD = np.load('current_phys_data/data_SD/test_X.npy', mmap_mode='r')
	X_train_SD = np.load('current_phys_data/data_SD/train_X.npy', mmap_mode='r')
	Y_test_SD = np.load('current_phys_data/data_SD/test_Y.npy', mmap_mode='r')
	Y_train_SD = np.load('current_phys_data/data_SD/train_Y.npy', mmap_mode='r')

	X_test = np.concatenate([X_test_DD, X_test_ND, X_test_SD], axis=0)
	X_train = np.concatenate([X_train_DD, X_train_ND, X_train_SD], axis=0)
	Y_test = np.concatenate([Y_test_DD, Y_test_ND, Y_test_SD], axis=0)
	Y_train = np.concatenate([Y_train_DD, Y_train_ND, Y_train_SD], axis=0)
	numbers_used = [1, 2, 3, 4]
	# print(Y_test)
	train_mask = np.isin(Y_train, numbers_used)
	test_mask = np.isin(Y_test, numbers_used)
	X_train, Y_train = X_train[train_mask], Y_train[train_mask]
	X_test, Y_test = X_test[test_mask], Y_test[test_mask]
	# print(X_test)
	# print(X_test.shape)
	# print(Y_test)

	X_train = X_train.astype('float32') / 255
	X_test = X_test.astype('float32') / 255


	[clust_train_data, clust_train_labels] = x_values(X_train, Y_train)
	print(clust_train_labels.shape, "train labels shape")
	print(clust_train_data, "clust train data")
	print(clust_train_data.shape, "clustdata shape")
	plt.show(block=False)
	[clust_test_data, clust_test_labels] = x_values(X_test, Y_test)
	classes_used = len(numbers_used)
	y_pred_kmeans = cluster(classes_used, clust_train_data, clust_test_data, clust_test_labels)
	extract_data(X_test, Y_test, y_pred_kmeans)
	plt.show()



if __name__ == "__main__":
	main()