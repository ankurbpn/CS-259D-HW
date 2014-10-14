from scipy.sparse import csr_matrix
import numpy as np
from sklearn.decomposition import RandomizedPCA
#Global parameters
#Index number for commands, will get updated as we encounter new commands
m = 0
M = 856 #Total number of commands in dataset

#Total number of sequences
NUM_SEQUENCES = 0

#Length of sequences
LENGTH_SEQUENCES = 100

#Total number of sequences for training
TOTAL_TRAINING_SEQUENCES = 5000*50/LENGTH_SEQUENCES

#Training sequences per user
TRAINING_SEQ = 5000/LENGTH_SEQUENCES

#Window size for co-occurence calculation/Scope
#Curently we assign 1 for each co-occurence within window irrespective of distance
SCOPE = 6

#Dimension of reduced space
N = 50

#Global map from command name to index number
command_to_index = {}
command_to_occ = {}

#Data matrix for storing training sequences in row major form
training_data = csr_matrix((TOTAL_TRAINING_SEQUENCES, M*M), dtype = float)

#Data vector for storing mean of the training sequences for centering
#meanData = csr_matrix((1, M*M), dtype = float)
#numVals = csr_matrix((1, M*M), dtype = float)

#Function to generate a co-occurence matrix vector of length M*M from command sequence
def generate_sequence_vector(sequence):
    global command_to_index
    global SCOPE

    #Currently adds one for every co-occurence, update to Gaussian?
    y = []
    data = []
    for i in range(len(sequence)):
        for j in range(SCOPE):
            if i + j + 1 < len(sequence):
                list_index = command_to_index.get(sequence[i])*M + command_to_index.get(sequence[j + i + 1])
                if list_index not in y:
                    y.append(list_index)
                    data.append(1)
                else:
                    ind = y.index(list_index)
                    data[ind] = data[ind] + 1
    return y, data




#To read through all the files for the 50 users and generate co-occurence matrices for training sequences for the 50 users
for i in range(50):
    userNo = i+1
    #print "User %d" % userNo
    file = open("User%d" % userNo, 'r')

    counter = 0
    #List to divide User data into sequences
    command_list = []
    seq_no = 0

    for line in file:
        cmd = line.strip('\n')
        counter = counter+1
        if counter > 5000:
            break
        command_list.append(cmd)
        if command_to_occ.has_key(cmd):
            command_to_occ[cmd] = command_to_occ[cmd]+1
        else:
            command_to_index[cmd] = m
            command_to_occ[cmd] = 0
            m = m+1

        if counter%LENGTH_SEQUENCES == 0:
            y, data = generate_sequence_vector(command_list)
            command_list = []
            x = [seq_no + i*TRAINING_SEQ]*len(y)
            seq_no = seq_no + 1
            training_data = training_data + csr_matrix((data, (x, y)), shape = (TOTAL_TRAINING_SEQUENCES, M*M))

print command_to_occ

#Centering training data
mean_data = training_data.mean(axis = 0)
np.expand_dims(mean_data, axis = 1)
ones_array = np.matrix([1]*TOTAL_TRAINING_SEQUENCES).T
training_data_centered = training_data - ones_array*mean_data
del training_data

pca = RandomizedPCA(n_components = N)
pca.fit(training_data_centered)
print pca.explained_variance_ratio_

reduced_feature_vectors = pca.transform(training_data_centered)
print pca.components_.shape
