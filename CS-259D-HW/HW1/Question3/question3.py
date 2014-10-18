#!/usr/bin/python
import numpy as np
import csv

TRAINING_DATA_FILENAME = "KeyboardData.csv"
TEST_DATA_FILENAME = "KeyboardTestData.csv"

CSV_COLUMN_SUBJECT_NAME = 0
CSV_COLUMN_KEY_DATA_START = 3

def load_data():
    cur_subject_values = None
    cur_subject = None

    subject_to_values_map = {}

    subject_to_means_map = {}

    # Load all the data, mapping every subject to their matrix of input data
    with open(TEST_DATA_FILENAME, 'rb') as csvfile:
        for row in csv.reader(csvfile):
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

    print subject_to_means_map

load_data()

