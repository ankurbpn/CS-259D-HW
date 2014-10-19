#!/usr/bin/python
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import empirical_covariance

TRAINING_DATA_FILENAME = "KeyboardData.csv"
TEST_DATA_FILENAME = "KeyboardTestData.csv"

CSV_COLUMN_SUBJECT_NAME = 0
CSV_COLUMN_KEY_DATA_START = 3

def load_data(file_name, training):
    cur_subject_values = None
    cur_subject = None

    subject_to_values_map = {}

    subject_to_means_map = {}

    # Load all the data, mapping every subject to their matrix of input data
    with open(file_name, 'rb') as csvfile:
        for row in csv.reader(csvfile):
	    if training:
		training = False
		continue
	    subject_name = row[CSV_COLUMN_SUBJECT_NAME]
            if cur_subject is None or subject_name != cur_subject:
                cur_subject = subject_name
                cur_subject_values = np.array([row[CSV_COLUMN_KEY_DATA_START:]], dtype=float)
                subject_to_values_map[cur_subject] = cur_subject_values
            
            else:
                row_as_array = np.array([row[CSV_COLUMN_KEY_DATA_START:]], dtype=float)
                cur_subject_values = np.append(cur_subject_values, row_as_array, axis=0)
                subject_to_values_map[cur_subject] = cur_subject_values

                
    # Collapse matrix of input data down to 1D vector of input data means for each subject
    for subject in subject_to_values_map.keys():
        values_matrix = subject_to_values_map[subject]
        means_of_values = values_matrix.mean(axis=0)
        subject_to_means_map[subject] = means_of_values

    return subject_to_means_map, subject_to_values_map

def pca_visual_analysis(subject_to_means, subject_to_values):
	full_matrix = None
	for key in subject_to_values.keys():
		#print subject_to_values[key].shape
		if full_matrix is None:
			full_matrix = subject_to_values[key]
		else:
			full_matrix = np.append(full_matrix, subject_to_values[key], axis = 0)
	pca = PCA(n_components=2)
	pca.fit(full_matrix)
	for key in subject_to_values.keys():
		plt.scatter(pca.transform(subject_to_values[key])[:, 0], pca.transform(subject_to_values[key])[:, 1], c = np.random.random_sample((3,)))
	plt.xlim(0,4)
	plt.ylim(-1, 1)
	plt.show()

def remove_redundant_features(subject_to_means, subject_to_values, list_of_redundant_columns):
	for key in subject_to_values.keys():
		subject_to_values[key] = np.delete(subject_to_values[key], list_of_redundant_columns, 1)
		subject_to_means[key] = np.delete(subject_to_means[key], list_of_redundant_columns, 1)
	return subject_to_means, subject_to_values

def covariances():
	subject_to_means, subject_to_values = load_data(TRAINING_DATA_FILENAME, True)
	subject_to_covariance = {}
	full_matrix = None
	for key in subject_to_values.keys():
		if full_matrix is None:	
			full_matrix = subject_to_values[key]
			subject_to_covariance[key] = empirical_covariance(subject_to_values[key])
			print subject_to_means[key]
			print subject_to_covariance[key]
		else:
			full_matrix = np.append(full_matrix, subject_to_values[key], axis = 0)
			subject_to_covariance[key] = empirical_covariance(subject_to_values[key])
	
	full_mean = full_matrix.mean(axis=0)
	full_covariance = empirical_covariance(full_matrix)
	print full_mean
	print full_covariance
	return subject_to_covariance, full_covariance, full_mean

def manhattan_distance(x, y):
	return np.sum(np.absolute(x-y))

def mahalanobis_distance(x, y, Sig):
	return np.sqrt((x-y)*np.linalg.pinv(Sig)*(x-y).T)
	
def mahalanobis_nearest_neighbor(x, Y, Sig):
	min_distance = 100000000000000
	for i in range(Y.shape[0]):
		temp = mahalanobis_distance(x, Y[i, :], Sig)
		if temp<min_distance:
			min_distance = temp
	return min_distance

def manhattan_nearest_neighbor(x, Y):
	min_distance = 10000000000000
	for i in range(Y.shape[0]):
		temp = manhattan_distance(x, Y[i, :])
		if temp < min_distance:
			min_distance = temp
	return min_distance

def output_manhattan_distances(training__subject_means):
	with open(DISTANCES_OUTPUT_FILENAME, 'wb') as output_file:
    		output_file_writer = csv.writer(output_file)
    	with open(TEST_DATA_FILENAME, 'rb') as csvfile:
        	for i, row in enumerate(csv.reader(csvfile)):
            		subject_nameect = row[CSV_COLUMN_SUBJECT_NAME]
            		data = np.array(row[CSV_COLUMN_KEY_DATA_START:], dtype=float)
            		means_for_subject = training__subject_means[subject]
            
            		manhattan_distance = scipy.spatial.distance.cityblock(means_for_subject, data)
            
            		output_file_writer.writerow([subject, manhattan_distance])


training__subject_to_means, training__subject_to_values = load_data(TRAINING_DATA_FILENAME, training=True)

pca_visual_analysis(training__subject_to_means, training__subject_to_values)

output_manhattan_distances(training__subject_to_means)
