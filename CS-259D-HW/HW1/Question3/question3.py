#!/usr/bin/python
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import random
import scipy
from sklearn.decomposition import PCA
from sklearn.covariance import empirical_covariance

TRAINING_DATA_FILENAME = "KeyboardData.csv"
TEST_DATA_FILENAME = "KeyboardTestData.csv"
DISTANCES_OUTPUT_FILENAME = "ManhattanDistances.csv"
ANSWERS_OUTPUT_FILENAME = "answer.csv"

CSV_COLUMN_SUBJECT_NAME = 0
CSV_COLUMN_KEY_DATA_START = 3

MANHATTAN = 0

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
                subject = row[CSV_COLUMN_SUBJECT_NAME]
                data = np.array(row[CSV_COLUMN_KEY_DATA_START:], dtype=float)
                means_for_subject = training__subject_means[subject]
                
                manhattan_distance = scipy.spatial.distance.cityblock(means_for_subject, data)
                
                output_file_writer.writerow([subject, manhattan_distance])


def get_manhattan_distances_from_test_file(training__subject_means):
    distances = []

    with open(TEST_DATA_FILENAME, 'rb') as csvfile:
        for row in csv.reader(csvfile):
            subject_nameect = row[CSV_COLUMN_SUBJECT_NAME]
            data = np.array(row[CSV_COLUMN_KEY_DATA_START:], dtype=float)
            means_for_subject = training__subject_means[subject_nameect]
            
            manhattan_distance = scipy.spatial.distance.cityblock(means_for_subject, data)

            distances.append( (subject_nameect, manhattan_distance) )

    return distances
            

def get_manhattan_distances(training_means, test_sequences):
    distances = []
    
    for subject in test_sequences:
        subject_mean = training_means[subject]
        for test_sequence in test_sequences[subject]:
            manhattan_distance = scipy.spatial.distance.cityblock(subject_mean, test_sequence)
            distances.append( (subject, manhattan_distance) )

    return distances

            		
def get_1_norm_data(training__subject_to_means, training__subject_to_values):
    subject_to_1_norm_means = {}
    subject_to_1_norm_stddvs = {}

    for subject in training__subject_to_means:
        features_means = training__subject_to_means[subject]
        features_values = training__subject_to_values[subject]

        norms = [np.linalg.norm(S_i - features_means, ord=1) for S_i in features_values]
        
        norms_mean = np.mean(norms)
        norms_stddv = np.std(norms)
        
        subject_to_1_norm_means[subject] = norms_mean
        subject_to_1_norm_stddvs[subject] = norms_stddv

    return (subject_to_1_norm_means, subject_to_1_norm_stddvs)


def get_distance_thresholds(subjects_to_values, subjects_to_means):
    subject_to_1_norms_means, subject_to_1_norms_stddvs = get_1_norm_data(training__subject_to_means, training__subject_to_values)

    subject_to_distance_thresholds = {}

    for subject in subject_to_1_norms_means:
        norm_mean = subject_to_1_norms_means[subject]
        norm_stddv = subject_to_1_norms_stddvs[subject]

        subject_to_distance_thresholds[subject] = norm_mean+norm_stddv

    return subject_to_distance_thresholds


def output_answers(input_labels_vector):
    with open(ANSWERS_OUTPUT_FILENAME, 'wb') as output_file:
        output_file_writer = csv.writer(output_file)
        for row in input_labels_vector:
            output_file_writer.writerow([row])


def label_test_input_manhattan(training__subject_to_means, training__subject_to_values):
    distance_thresholds = get_distance_thresholds(training__subject_to_values, training__subject_to_means)
    
    manhattan_distances = get_manhattan_distances_from_test_file(training__subject_to_means)
    
    input_labels_vector = [0 if distance <= distance_thresholds[subject] else 1 for (subject, distance) in manhattan_distances]
    
    output_answers(input_labels_vector)



def get_validation_and_training_sequences(training__subject_to_values):
    validation_sequences = {}
    for subject in training__subject_to_values:
        training_sequences = training__subject_to_values[subject]
        
        num_training_sequences = len(training_sequences)
        sample_size = int(math.floor(0.2 * num_training_sequences))
        
        # Pick random indeces of training sequences to convert to validation sequences
        training_sequences_to_remove = random.sample(xrange(num_training_sequences), sample_size)

        # Copy over the validation sequences
        validation_sequences[subject] = training_sequences.take(training_sequences_to_remove, axis=0)

        # Remove the training sequences
        training__subject_to_values[subject] = np.delete(training_sequences, training_sequences_to_remove, 0)

    return validation_sequences, training__subject_to_values


def measure_performance(training__subject_to_values, algorithm=MANHATTAN):
    validation_seqs, training_seqs = get_validation_and_training_sequences(training__subject_to_values)

    training_means = {}
    for subject in training_seqs:
        training_means[subject] = np.mean(training_seqs[subject])

    input_labels_vector = None

    if algorithm == MANHATTAN:
        distance_thresholds = get_distance_thresholds(training_seqs, training_means)
        manhattan_distances = get_manhattan_distances(training_means, validation_seqs)
        input_labels_vector = [0 if distance <= distance_thresholds[subject] else 1 for (subject, distance) in manhattan_distances]

    # Number of incorrect classifications is the number of passwords rejected from the validation sequences
    # The number of rejected passwords is just the sum of each element in the input labels vector, as every
    # element is an indicator variable
    percent_false_negative = float(np.sum(input_labels_vector)) / len(input_labels_vector)
    percent_accuracy = 1 - percent_false_negative

    return (percent_false_negative, percent_accuracy)
    
training__subject_to_means, training__subject_to_values = load_data(TRAINING_DATA_FILENAME, training=True)

#pca_visual_analysis(training__subject_to_means, training__subject_to_values)

label_test_input_manhattan(training__subject_to_means, training__subject_to_values)

(manhattan_false_negative_rate, manhattan_accuracy_rate) = measure_performance(training__subject_to_values, algorithm=MANHATTAN)

print "Manhattan false negative rate: %f" % manhattan_false_negative_rate
print "Manhattan accuracy rate: %f" % manhattan_accuracy_rate
