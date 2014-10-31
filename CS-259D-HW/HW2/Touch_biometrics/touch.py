import numpy as np
from sklearn import *
import csv
##import matplotlib.pyplot as plt
##import pylab as py
import math 
from heapq import *
import multiprocessing

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
	return MIUF/HU
 
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
	return cor

def print_rel_entropy():
	dic = get_feature_dict()
	for i in range(32):
		print dic[i]
		rel_entropy(i)

##This function selects the list of features based on maximum mutual information with the user from the 30 old features
def select_feature_1():
	feat = []
	lis = []
	dic = get_feature_dict()
	for i in range(30):
		temp = rel_entropy(i)
		heappush(feat, (1/temp, i))
	
	for j in range(10):
		temp = heappop(feat)[1]
		lis.append(temp)
		print dic[temp]
	return lis

##This function selects the list of features based on mini max correlation with the features currently in the list - the first feature is the one with the maximum relative entropy

def select_feature_2():
	#First feature is mid stroke pressure which is feature number 26
	feat = [26]
	cor = correlation()
	dic = get_feature_dict()
	print dic[feat[0]]
	for k in range(9):
		min_index = -1
		min = 1
		for i in range(30):
			temp = np.amax(cor[i, feat])
			if temp < min:
				min_index = i
				min = temp
		feat.append(min_index)
		print dic[min_index]
	
	return feat

##This function trains a tree for multi-class classification and chooses the features with highest importance for classification
def select_feature_3():
	user, features = read_features()
	dic = get_feature_dict()
	tree = ensemble.ExtraTreesClassifier()
	tree.fit(features[:, range(30)], user)
	imp = tree.feature_importances_
	lis = []
	feat = []
	for i in range(30):
		heappush(lis, (1/imp[i], i))
	for i in range(10):
		temp = heappop(lis)[1]
		feat.append(temp)
		print dic[temp]
	return feat

##Use L-1 norm with a linear classifer to select most important features
def select_feature_4():
	pen_factor = [.1, .5, 1, 5, 10, 50, 100, 500, 1000]
	lis = multiprocessing.Pool(len(pen_factor)).map(get_selected_features, pen_factor)	
	for item in lis:
		print item.shape

def get_selected_features(p):
	user, features = read_features()
	select = linear_model.LogisticRegression(penalty='l1', C=p, dual=False)
	select.fit(features[:, range(30)], user)
	feat = select.transform(np.matrix(range(30)))
	return feat

def get_F1_score():
	user, features = read_features()
	user_set = set(list(np.squeeze(user)))
	##Select features
	feat = select_feature_1()
	print user_set
	F1 = []
	for u in user_set:
		y = 0*user
		for i in range(user.shape[0]):
			if user[i]!=u:
				y[i]=1
		##Use SVM Classifier
		model = svm.SVC(kernel = 'rbf')
		##Use Log reg classifier
		#model = linear_model.LogisticRegression
		F1.append(np.mean(cross_validation.cross_val_score(model, features[:, feat], y, cv=10)))
	print 'F1 score is ',np.mean(np.array(F1)) 
					

##print_rel_entropy()
##correlation()
##select_feature_1()
##select_feature_2()
##select_feature_3()
##select_feature_4()

get_F1_score()

