import sys
import numpy as np

data = np.matrix([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

def normalize(data):
	m, n = data.shape[0], data.shape[1]
	data_normal = np.zeros((m, n))

	for i in range(n):
		feature_mean = np.mean(data[:, i])
		feature = data[:, i] - feature_mean
		sd = np.sqrt(np.dot(feature.T, feature))[0, 0]
		data_normal[:, i] = feature.flatten() / sd

	return data_normal

def calculate_covariance_matrix(data):
	data_count = data.shape[0]
	feature_count = data.shape[1]
	covariance_matrix = np.zeros((feature_count, feature_count))

	for i in range(feature_count):
		for j in range(i, feature_count):
			feature_1 = data[:, i]
			feature_2 = data[:, j]

			feature_1_mean = np.mean(feature_1)
			feature_2_mean = np.mean(feature_2)

			covariance_matrix[i, j] = np.dot((feature_1 - feature_1_mean).T, (feature_2 - feature_2_mean)) / (data_count - 1)
			covariance_matrix[j, i] = np.dot((feature_1 - feature_1_mean).T, (feature_2 - feature_2_mean)) / (data_count - 1)

	return covariance_matrix

def PCA(data, k=None):
	if k is None:
		k = data.shape[1]

	data_normal = normalize(data)

	covariance_matrix = calculate_covariance_matrix(data_normal)
	eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)	# each column in eignvectors refers to a feature

	feature_maintain_count = 0

	eigenvalues_sorted = list(reversed(np.argsort(eigenvalues)))
	eigenvalues_selected = eigenvalues_sorted[: k]
	
	features_selected = eigenvectors[:, eigenvalues_selected]
	return np.dot(data_normal, features_selected)

data_compressed = PCA(data, 1)
print(data_compressed)









