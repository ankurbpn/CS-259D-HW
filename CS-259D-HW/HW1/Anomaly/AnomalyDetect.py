from scipy.sparse import csr_matrix
import numpy as np
from sklearn.decomposition import RandomizedPCA
#Global parameters
#Index number for commands, will get updated as we encounter new commands
m = 0
M = 635 #Total number of commands in training dataset

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
N = 48

#Co-occurence Threshold
THRESHOLD = 0.1

 
#Global map from command name to index number
command_to_index = {}
command_to_occ = {}

#Data matrix for storing training sequences in row major form
training_data =csr_matrix((TOTAL_TRAINING_SEQUENCES, M*M), dtype = float)

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


##Add function that takes in a sequence feature vector and the pca components as input and outputs the corresponding layered network matrix
def get_layered_network(X, pca):
	temp_matrix = csr_matrix((N, N))
	layered_matrix = temp_matrix.setdiag(X)*pca
	#Removing elements below the threshold from the positive layered matrix
	positive_bool_matrix = layered_matrix > THRESHOLD
	negative_bool_matrix = layered_matrix < -THRESHOLD
	return positive_bool_network, negative_bool_network

#Add function to compare two layered networks and output a similarity score, based on this decide a threshold.
def layered_network_similarity(X, Y):
	return 2*np.sum(logical_and(X, Y))/(np.sum(X)+np.sum(Y))


#To read through all the files for the 50 users and generate co-occurence matrices for training sequences for the 50 users
def get_pca_decompositions():
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
	    NUM_SEQUENCES = NUM_SEQUENCES + seq_no

	print 'Number of sequences = ', NUM_SEQUENCES
	print 'Number of different commands = ', m, len(command_to_index.keys()), len(command_to_occ.keys())

	index = M* command_to_index['rm'] +  command_to_index['ls']
	print 'Number of co-occurences of rm with ls for user 1 and 2 for the first 5 sequences are:'
	print 'User 1: ', training_data[0, index], training_data[1, index],training_data[2, index],training_data[3, index],training_data[4, index] 
	print 'User 2: ', training_data[1*TRAINING_SEQ, index],training_data[1+TRAINING_SEQ, index],training_data[2+TRAINING_SEQ, index],training_data[3+TRAINING_SEQ, index],training_data[4+TRAINING_SEQ, index]


	#Centering training data
	mean_data = training_data.mean(axis = 0)
	np.expand_dims(mean_data, axis = 1)
	ones_array = np.matrix([1]*TOTAL_TRAINING_SEQUENCES).T
	training_data_centered = training_data - ones_array*mean_data
	del training_data
	print 'We chose exact mean centering (destroys sparsity of the training matrix)'
	index = M* command_to_index['rm'] +  command_to_index['ls']
	print 'Centered correlation of rm with ls for user 1 and 2 for the first 5 sequences are:'
	print 'User 1: ', training_data_centered[0, index], training_data_centered[1, index],training_data_centered[2, index],training_data_centered[3, index],training_data_centered[4, index]
	print 'User 2: ', training_data_centered[1*TRAINING_SEQ, index],training_data_centered[1+TRAINING_SEQ, index],training_data_centered[2+TRAINING_SEQ, index],training_data_centered[3+TRAINING_SEQ, index],training_data_centered[4+TRAINING_SEQ, index]

	pca = RandomizedPCA(n_components = N)
	pca.fit(training_data_centered)
	contribution_ratio = np.sqrt(pca.explained_variance_ratio_)
	print 'The unnormalized contribution ratio is', contribution_ratio
	net_contribution = np.sum(contribution_ratio)
	contribution_ratio = contribution_ratio/net_contribution
	print 'the normalized contribution ratio is', contribution_ratio

	reduced_feature_vectors = pca.transform(training_data_centered)
	print 'pca components shape', pca.components_.shape

	#Code to extract the feature vectors for all sequences for users 1 to 50 
	features_1_to_50 = pca.transform(training_data_centered)

	del training_data_centered
	
	np.savetxt("mean.csv", mean_data, delimiter=",")
	np.savetxt("pca.csv", pca.components_, delimiter=",")
	for i in range(50):
		np.savetxt("feature%d.csv" (i+1), features_1_to_50[(TRAINING_SEQ*i):(TRAINING_SEQ*(i+1)),:], delimiter=",")
 
#Code to convert all these feature vectors into layered networks (N layered networks per sequence) and using them to find outputs for sequences 
#from the test set
def find_malicious_users(SIM_THRESHOLD=0.5):
	
	pca = np.readtxt(open("pca.csv","rb"),delimiter=",")
	mean_data = np.readtxt(open("mean.csv","rb"),delimiter=",")
	
	test_data_malicious = np.zeros(shape = (100, 50))
	
	#To read through all the files for the 50 users and generate co-occurence matrices for training sequences for the 50 users
	for i in range(50):
	    training_features = np.readtxt("feature%d.csv", "rb", delimiter = "," )
	    reference_layered_network_pos = np.matrix(shape = (TRAINING_SEQ*N, M*M))
	    reference_layered_network_neg = np.matrix(shape = (TRAINING_SEQ*N, M*M))
 	for j in range(TRAINING_SEQ):
	    reference_layered_network_pos[(N*j):(N*(j+1)-1), :], reference_layered_network_neg[(N*j):(N*(j+1)-1), :] =  get_layered_network(training_features[j, :], pca)
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
		if counter <= 5000:
		    continue
		command_list.append(cmd)
		
		if counter%LENGTH_SEQUENCES == 0:
		    y, data = generate_sequence_vector(command_list)
		    command_list = []
		    x = [0]*len(y)
		    seq_no = seq_no + 1
		    sequence_matrix = csr_matrix((data, (x, y)), shape = (1, M*M)) - mean_data
		    features = sequence_matrix*pca.T
		    sequence_layered_network_pos, sequence_layered_network_neg = get_layered_network(features, pca)
		    max_similarity = 0.0
		    for j in range(TRAINING_SEQ):
			sim = 0.0
			for k in range(N):
				sim = sim + (get_layered_network_similarity(sequence_layered_network_pos[k, :], reference_layered_network_pos[j*N+k, :]) + get_layered_network_similarity(sequence_layered_network_neg[k, :], reference_layered_network_neg[j*N+k, :]))/2
			if max_similarity < sim:
				max_similarity = sim
		    if max_similarity < SIM_THRESHOLD:
			test_data_malicious[seq_no-1, i] = 1
	np.savetxt("results.csv", test_data_malicious, delimiter=",")










