import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

def distance(centroid, data):
	return np.sqrt(np.sum(np.power((centroid - data), 2)))

def select_centroid(cluster, c):
	if c == 1:
		return np.mean(cluster, axis=0)
	else:
		axis_min = np.min(cluster, axis=0)
		axis_max = np.max(cluster, axis=0)
		step_size = (axis_max - axis_min) / c

		return [((axis_min + i * step_size) + ((axis_min + i * step_size) + step_size)) / 2.0 for i in range(c)]
		# for i in range(c):
		# 	cur_min = axis_min + i * step_size
		# 	cur_max = cur_min + step_size
		# 	cent_cand = (cur_max + cur_min) / 2.0

def k_means(dataset, centroid_count):
	centroid = select_centroid(dataset, centroid_count)
	data_count = dataset.shape[0]
	clusters = np.matrix(np.zeros((data_count, 2)))
	change = True
	i = 0

	while change is True:
		change = False
		for data_index in range(data_count):
			min_distance = np.inf
			cluster = clusters[data_index, 0]

			for index, cent in enumerate(centroid):
				cur_distance = distance(cent, dataset[data_index, :])
				if cur_distance < min_distance:
					min_distance = cur_distance
					cluster = index
			if cluster != clusters[data_index, 0] or i == 0:
				if cluster == -1:
					raise Exception('error happens')
				clusters[data_index, :] = cluster, min_distance ** 2
				change = True
		for index, cent in enumerate(centroid):
			cur_cluster = dataset[np.where(clusters[:, 0] == index)[0]]		# important method, which indexing matrix by conditions
			centroid_new = np.mean(cur_cluster, axis=0)
			centroid[index] = centroid_new
		i += 1
	return clusters, centroid

def bisecting_kmeans(dataset, k):
	data_count = dataset.shape[0]
	clusters = np.zeros((data_count, 2))
	centroid_init = select_centroid(data_count, 2)
	centList = [centroid_init]

	for data_index in range(data_count):
		clusters[data_index, 1] = distance(centroid_init, dataset[data_index, :]) ** 2

	while len(centList) < k:
		lowestSSE = np.inf
		for i in range(len(centList)):
			cur_cluster = dataset[np.where(clusters[:, 0] == i)[0]]
			bi_cluster, centroid = k_means(cur_cluster, 2)

			err_new_c = np.sum(bi_cluster[:, 1])
			err_old_c = np.sum(clusters[np.where(clusters[:, 0] == i)[0]][:, 1])
			
			if (err_new_c + err_old_c < lowestSSE):
				best_c_index = i
				best_centroid = centroid
				best_cluster = bi_cluster.copy()
				lowestSSE = err_new_c + err_old_c
		best_cluster[np.where(best_cluster[:, 0] == 1)[0], 0] = len(centList)		# attention here, do not use [:, 0] to assign number
		best_cluster[np.where(best_cluster[:, 0] == 0)[0], 0]= best_c_index
		clusters[np.where(clusters[:, 0] == best_c_index)[0]] = best_cluster

		centList[best_c_index] = best_centroid[0]
		centList.append(best_centroid[-1])

	return centList, clusters

if __name__ == '__main__':
	centers = [[1, 1], [-1, -1], [1, -1]]
	X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)
	
	test_data = np.matrix([[2, 4], [4, 3], [6, 8]])
	# result, _ = k_means(X, 3)
	_, result = bisecting_kmeans(X, 3)

	fig = plt.figure(figsize=(8, 3))
	fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
	colors = ['#4EACC5', '#FF9C34', '#4E9A06']

	ax = fig.add_subplot(1, 3, 1)
	for i, col in zip(range(3), colors):
	    ax.plot(X[np.where(result[:, 0] == i)[0]][:, 0], X[np.where(result[:, 0] == i)[0]][:, 1], 'w', markerfacecolor=col, marker='.')
	ax.set_title('KMeans')
	ax.set_xticks(())
	ax.set_yticks(())
	plt.show()











