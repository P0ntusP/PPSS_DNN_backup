
def autoencoder_pythia(sigma_1, sigma_2):
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


	#extracting data from pythia files

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
	#print(Y_test)
	train_mask = np.isin(Y_train, numbers_used)
	test_mask = np.isin(Y_test, numbers_used)
	X_train, Y_train = X_train[train_mask], Y_train[train_mask]
	X_test, Y_test = X_test[test_mask], Y_test[test_mask]
	#print(X_test)
	#print(X_test.shape)
	X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
	X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))
	#print(Y_test)
	# Parameters for blurring
	#print(X_test)
	X_plot = X_test
	sigma_1 =100 #y_ish
	sigma_2 =50 #x_ish
	X_train = gaussian_filter(X_train, sigma=[sigma_1, sigma_2], order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
	X_test = gaussian_filter(X_test, sigma=[sigma_1, sigma_2], order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
	X_train = X_train.astype('float32') / 255
	X_test = X_test.astype('float32') / 255
	#print(X_test)
	#print(X_test.shape)

	# hyper parameters

	batch_size = 256
	epochs = 20
	bottle_dim = 4
	# Neural net
	########################################################################################################################
	#Input layer
	input_img = Input(shape=(784,))
	# architecture
	encoded = Dense(units=400, activation='relu')(input_img)
	encoded = Dense(units=100, activation='relu')(encoded)
	encoded = Dense(units=29, activation='relu')(encoded)
	encoded = Dense(units=bottle_dim, activation='linear')(encoded)

	decoded = Dense(units=29, activation='relu')(encoded)
	decoded = Dense(units=100, activation='relu')(decoded)
	decoded = Dense(units=400, activation='relu')(decoded)
	# output layer
	decoded = Dense(units=784, activation='sigmoid')(decoded)
	##################################################################################################################
	autoencoder = Model(input_img, decoded)

	encoder = Model(input_img, encoded)

	autoencoder.summary()

	encoder.summary()

	autoencoder.compile(optimizer='SGD', loss="mean_squared_logarithmic_error", metrics=['accuracy'])
	history = autoencoder.fit(X_train, X_train,
	                          epochs=epochs,
	                          batch_size=batch_size,
	                          shuffle=True,
	                          validation_data=(X_test, X_test))





	# summarize history for loss
	def loss_plot():
		print(history)
		plt.plot(history.history['loss'])

		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train loss'], loc='upper left')
		plt.show(block=False)
	#loss_plot()



	#Clustering
	def clustering():
		encoded_imgs_test = encoder.predict(X_test)
		kmeans = KMeans(n_clusters=len(numbers_used), n_init=40).fit(encoded_imgs_test)
		y_pred_kmeans = kmeans.predict(encoded_imgs_test)
		y_pred_kmeans = y_pred_kmeans

		#Scoring
		score = sklearn.metrics.rand_score(Y_test, y_pred_kmeans)






		print("score is: ", score)
		# Confusion matrix
		from sklearn.metrics import confusion_matrix

		cm = confusion_matrix(y_true=Y_test, y_pred=y_pred_kmeans)
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
		#sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
		acc = np.trace(cm2) / np.sum(cm2)
		print("accuracy is: ", acc)
		return y_pred_kmeans

	pred_kmeans = clustering()

	def extract_data(Y_test, y_pred_kmeans):
		#X_test = X_plot
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
	#extract_data(Y_test, pred_kmeans)

	# Plots the figueres
	def plot_numbers():
		# Prepeares the images for plotting and clustering
		encoded_imgs = encoder.predict(X_test)
		predicted = autoencoder.predict(X_test)
		plt.figure(figsize=(80, 4))
		for i in range(20):
			# display original images
			ax = plt.subplot(3, 20, i + 1)
			plt.imshow(X_test[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# display encoded images
			ax = plt.subplot(3, 20, i + 1 + 20)
			plt.imshow(encoded_imgs[i].reshape(bottle_dim, 1))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			# display reconstructed images
			ax = plt.subplot(3, 20, 2 * 20 + i + 1)
			plt.imshow(predicted[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		plt.show()

	plt.show()
	#plot_numbers()

autoencoder_pythia(0, 0)
