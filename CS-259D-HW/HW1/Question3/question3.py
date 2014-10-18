#!/usr/bin/python
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

def pca_visual_analysis():
	subject_to_means, subject_to_values = load_data(TRAINING_DATA_FILENAME, True)
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
pca_visual_analysis()

