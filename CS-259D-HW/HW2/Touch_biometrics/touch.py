import numpy as np
from sklearn import *
import csv
import matplotlib.pyplot as plt
import pylab as py
import math 

def get_feature_dict():
	featureStr = {}
	featureStr[0] = 'inter-stroke time'

	featureStr[1] = 'stroke duration'

	featureStr[2] = 'start $x$'

	featureStr[3] = 'start $y$'

	featureStr[4] = 'stop $x$'

	featureStr[5] = 'stop $y$'

	featureStr[6] = 'direct end-to-end distance'

	featureStr[7] = ' mean resultant lenght'

	featureStr[8] = 'up/down/left/right flag'

	featureStr[9] = 'direction of end-to-end line'

	featureStr[10] = '20\%-perc. pairwise velocity'

	featureStr[11] = '50\%-perc. pairwise velocity'

	featureStr[12] = '80\%-perc. pairwise velocity'

	featureStr[13] = '20\%-perc. pairwise acc'

	featureStr[14] = '50\%-perc. pairwise acc'

	featureStr[15] = '80\%-perc. pairwise acc'

	featureStr[16] = 'median velocity at last 3 pts'

	featureStr[17] = 'largest deviation from end-to-end line'

	featureStr[18] = '20\%-perc. dev. from end-to-end line'

	featureStr[19] = '50\%-perc. dev. from end-to-end line'

	featureStr[20] = '80\%-perc. dev. from end-to-end line'

	featureStr[21] = 'average direction'

	featureStr[22] = 'length of trajectory'

	featureStr[23] = 'ratio end-to-end dist and length of trajectory'

	featureStr[24] = 'average velocity'

	featureStr[25] = 'median acceleration at first 5 points'

	featureStr[26] = 'mid-stroke pressure'

	featureStr[27] = 'mid-stroke area covered'

	featureStr[28] = 'mid-stroke finger orientation'

	featureStr[29] = 'phone orientation'

	featureStr[30] = 'beginning to mid stroke pressure variation'

	featureStr[31] = 'median area variation'
	return featureStr

def read_features():
	data_matrix = np.loadtxt(open('featMat.csv', 'rb'), delimiter = ',')
	#print data_matrix.shape
	data_matrix = data_matrix[~np.isnan(data_matrix).any(axis=1)]
	feature_indices = range(2, 12)
	feature_indices.extend(range(13, 36))
	feature_indices.remove(32)
	#print feature_indices
	features = data_matrix[:, feature_indices]
	#print features.shape
	user = data_matrix[:, [0]]
	#print user
	return user, features

def rel_entropy(feature_index):
	user, features = read_features()
	user = np.squeeze(user)
	feature = np.squeeze(features[:, feature_index])
	min_value = np.percentile(feature, 10)
	max_value = np.percentile(feature, 90)
	num_buckets = 50
	#print len(user), len(feature)
	bucketed_feature = 0*feature
	step = (max_value - min_value)/(num_buckets)
	for i in range(len(feature)):
		val = min_value
		if feature[i]<val:
			bucketed_feature[i] = 0
			continue
		for j in range(num_buckets):
			val += step
			if feature[i]<val:
				bucketed_feature[i] = j+1
				break
		if feature[i]>val:
			bucketed_feature[i] = num_buckets-1
	joint_count = {}
	user_count = {}
	feature_count = {}
	total_count = 1.0*len(user)
	##Getting feature and user counts/probabilities
	for i in range(len(user)):
		if (user[i], bucketed_feature[i]) in joint_count.keys():
			joint_count[(user[i], bucketed_feature[i])] += 1
			user_count[user[i]]+=1
			feature_count[bucketed_feature[i]]+=1
		else:
			joint_count[(user[i], bucketed_feature[i])] = 1
			if user[i] in user_count.keys():
				user_count[user[i]]+=1
			else:
				user_count[user[i]]=1
			if bucketed_feature[i] in feature_count.keys():
				feature_count[bucketed_feature[i]]+=1
			else:
				feature_count[bucketed_feature[i]]=1
	
	##Computing mutual information and entopy
	MIUF = 0
	HU = 0
	for key in joint_count.keys():
		if joint_count[key]>0:
			MIUF += joint_count[key]*math.log(joint_count[key]*total_count/(user_count[key[0]]*feature_count[key[1]]))
	for key in user_count.keys():
		if user_count[key]>0:
			HU += user_count[key]*math.log(total_count/user_count[key])
	print 'Relative entropy is ', MIUF/HU
 
def correlation():
	user, features = read_features()
	user = np.squeeze(user)
	dic = get_feature_dict()
	for key in dic.keys():
		print key, dic[key]
	cov = covariance.EmpiricalCovariance() 
	cov.fit(features)
	cor = cov.covariance_
	for i in range(cor.shape[0]):
		for j in range(cor.shape[1]):
			if i != j:
				cor[i, j] = cor[i, j]/math.sqrt(cor[i, i]*cor[j, j])

	for i in range(cor.shape[0]):
		cor[i, i] = 1
	print dic[30]
	print cor[30, :]
	print dic[31]
	print cor[31, :]

def print_rel_entropy():
	dic = get_feature_dict()
	for i in range(32):
		print dic[i]
		rel_entropy(i)

print_rel_entropy()
correlation()
