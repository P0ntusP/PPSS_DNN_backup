#Clustering
def clustering():
	encoded_imgs = encoder.predict(X_test)
	kmeans = KMeans(n_clusters=10, n_init=100)
	y_pred_kmeans = kmeans.fit_predict(encoded_imgs)
	y_pred_kmeans[:10]
	#Scoring
	score = sklearn.metrics.rand_score(Y_test, y_pred_kmeans)
	print("score is: ", score)
clustering()