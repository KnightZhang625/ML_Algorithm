# startup for boosting algorithm, implement decision tree by ID3

###############################################	 calculate entropy ###############################################
import sys
from math import log

def extract_class(data):
	all_class = {}
	for x, y in data:
		if y not in all_class.keys():
			all_class[y] = [x]
		else:
			all_class[y].append(x)
	return all_class

def empirical_entropy(data):
	'''
		data : [([x1, x2, ... , xn], y), ...]
	'''
	# 1. create a dict, which allocating each data to its corresponding class
	### rewrite above
	# all_class = {}
	# for x, y in data:
	# 	if y not in all_class.keys():
	# 		all_class[y] = [x]
	# 	else:
	# 		all_class[y].append(x)
	###
	all_class = extract_class(data)
	# 2. calculate the empirical entropy
	h_d = 0
	d = len(data)
	for c, xs in all_class.items():
		h_d -= (len(xs) / d * log(len(xs) / d, 2))
	return h_d

def empirical_conditional_entropy(data, feature):
	# count the values of given feature
	feature_type = list(set([x[feature] for x, _ in data]))

	hd_a = 0
	d = len(data)
	for fe in feature_type:
		fe_count = 0
		fe_data = []
		for x, y in data:
			if x[feature] is fe:
				fe_count += 1
				fe_data.append((x, y))
		fe_empirical_entropy = empirical_entropy(fe_data)
		hd_a += (fe_count / d * fe_empirical_entropy)
	return hd_a
##################################################################################################################

################################################ ID3 Algorithm ###################################################
def ID3(data, feature, threshold):
	all_class = extract_class(data)
	classes = list(all_class.keys())

	if len(classes) == 1:
		return classes[-1]
	elif len(feature) == 0:
		all_class_sorted = sorted(all_class.items(), key=lambda x : len(x[1]), reverse=True)
		return all_class_sorted[0][0]
	else:
		g = []
		h_d = empirical_entropy(data)
		for index, fe in enumerate(feature):
			hd_a = empirical_conditional_entropy(data, index)
			g.append(h_d - hd_a)
		max_g = max(g)
		if max_g < threshold:
			all_class_sorted = sorted(all_class.items(), key=lambda x : len(x[1]), reverse=True)
			return all_class_sorted[0][0]
		else:
			idx_max = g.index(max_g)
			fe_selected = feature[idx_max]
			feature_type = list(set([x[idx_max] for x, _ in data]))

			feature.remove(fe_selected)
			dic_temp = {fe_selected : {}}
			for fe_v in feature_type:
				data_fe = []
				for d in data:
					if d[0][idx_max] == fe_v:
						d[0].pop(idx_max)
						data_fe.append(d)
				dic_temp[fe_selected][fe_v] = ID3(data_fe, feature, threshold)
			return dic_temp
##################################################################################################################

if __name__ == '__main__':
	data = [(['young', 'no', 'no', 'normal'], 0), (['young', 'no', 'no', 'good'], 0), (['young', 'yes', 'no', 'good'], 1),
			(['young', 'yes', 'yes', 'normal'], 1), (['young', 'no', 'no', 'normal'], 0), (['adult', 'no', 'no', 'normal'], 0),
			(['adult', 'no', 'no', 'good'], 0), (['adult', 'yes', 'yes', 'good'], 1), (['adult', 'no', 'yes', 'excellent'], 1),
			(['adult', 'no', 'yes', 'excellent'], 1), (['old', 'no', 'yes', 'excellent'], 1), (['old', 'no', 'yes', 'good'], 1),
			(['old', 'yes', 'no', 'good'], 1), (['old', 'yes', 'no', 'excellent'], 1),  (['old', 'no', 'no', 'normal'], 0)]      
	
	# h_d = empirical_entropy(data)
	# hd_a = empirical_conditional_entropy(data, 0)

	t = ID3(data, ['age', 'job', 'house', 'credit'], 0.001)
	print(t)












