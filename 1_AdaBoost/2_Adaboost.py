# this is the second tutorial of Adaboost
# we have introduced the decision tree algorithm,
# however, this time, we will use a naive tree which makes its decision based on only one feature

import sys
import time
import numpy as np
from math import log

def predict(data, fe, threshold, com):
	prediction = np.ones((data.shape[0], 1))
	if com == 'lt':
		prediction[data[:, fe] <= threshold] = - 1.0
	else:
		prediction[data[:, fe] > threshold] = -1.0
	return prediction
	

def buildTree(data, classLabels, weight):
	'''
		data : data_count * feature_count
		classLabels : list
		weight : 1 * data_count
	'''
	data_count, feature_count = data.shape
	classLabels = classLabels.T
	min_error = np.inf
	steps = 10.0
	best_tree = {}
	best_prediction = np.matrix(np.zeros((data_count, 1)))
	
	for fe in range(feature_count):
		fe_min = data[:, fe].min()
		fe_max = data[:, fe].max()
		pre_step = (fe_max - fe_min) / steps
		
		for s in range(-1, int(steps)+1):
			for com in ['lt', 'gt']:
				threshold = fe_min + pre_step * float(s)
				prediction = predict(data, fe, threshold, com)
				error_matrix = np.matrix(np.ones((data_count, 1)))
				error_matrix[error_matrix == classLabels] = 0
				weightedError = float((weight * error_matrix))
				if weightedError < min_error:
					min_error = weightedError
					best_prediction = prediction.copy()
					best_tree['fe'] = fe
					best_tree['threshold'] = threshold
					best_tree['com'] = com
	return best_tree, min_error, best_prediction

def adaboost(data, classLabels, init_weight, iteration=10):
	naive_tree = []
	data_count = data.shape[0]
	classLabels = np.matrix(classLabels)
	weight = init_weight
	aggPrediction = np.matrix(np.zeros((data_count, 1)))

	for epoch in range(iteration):
		best_tree, min_error, best_prediction = buildTree(data, classLabels, weight)
		alpha = float(0.5 * log((1.0 - min_error) / max(min_error, 1e-16)))
		best_tree['alpha'] = alpha

		# calculate weights for data
		expon = np.exp((- np.multiply(best_prediction, classLabels.T)) * alpha)
		numerator = np.multiply(weight.T, expon)
		denominator = numerator.sum()
		weight = (numerator / denominator).T

		# use new alpha to predict, and count the error
		aggPrediction += alpha * best_prediction
		error = np.multiply(np.sign(aggPrediction) != classLabels.T, np.ones((data_count, 1)))
		errorRate = error.sum() / data_count
		print('the %dth inter\'s error is : %.2f'%(epoch, errorRate))
		if errorRate == 0.0:
			break
	return naive_tree

if __name__ == '__main__':
	# build some sample data
	dataMat = np.matrix([[1., 2.1], 
						 [2., 1.1],
						 [1.3, 1.],
						 [1.,  1.],
						 [2.,  1.,]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

	init_weight = np.matrix([1/len(classLabels) for c in classLabels])

	adaboost(dataMat, classLabels, init_weight)











