from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.decomposition import RandomizedPCA
import csv

def get_global_vars():
	#Global parameters
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

	THRESHOLD = 0.5

	#Dimension of reduced space
	N = 48
	#Data vector for storing mean of the training sequences for centering
	#meanData = csr_matrix((1, M*M), dtype = float)
	#numVals = csr_matrix((1, M*M), dtype = float)
	return M, NUM_SEQUENCES,LENGTH_SEQUENCES, TOTAL_TRAINING_SEQUENCES, TRAINING_SEQ, SCOPE, THRESHOLD, N

#Function to generate a co-occurence matrix vector of length M*M from command sequence
def generate_sequence_vector(sequence, command_to_index):
    M, NUM_SEQUENCES, LENGTH_SEQUENCES, TOTAL_TRAINING_SEQUENCES,TRAINING_SEQ, SCOPE, THRESHOLD, N = get_global_vars()
    #Currently adds one for every co-occurence, update to Gaussian?
    y = []
    data = []
    for i in range(len(sequence)):
	for j in range(SCOPE):
	    if i + j + 1 < len(sequence):
		if command_to_index.has_key(sequence[i]):
			if command_to_index.has_key(sequence[i+j+1]):
				list_index = command_to_index.get(sequence[i])*M + command_to_index.get(sequence[j + i + 1])
				if list_index not in y:
		    			y.append(list_index)
		    			data.append(1.0)
				else:
		    			ind = y.index(list_index)
		    			data[ind] = data[ind] + 1.0
    return y, data


##Add function that takes in a sequence feature vector and the pca components as input and outputs the corresponding layered network matrix
def get_layered_network(X, pca):
	M, NUM_SEQUENCES, LENGTH_SEQUENCES, TOTAL_TRAINING_SEQUENCES,TRAINING_SEQ, SCOPE, THRESHOLD, N = get_global_vars()
        temp_matrix = csr_matrix((N, N), dtype = float)
	#print temp_matrix.shape, X.shape
	temp_matrix.setdiag(X)
	layered_matrix = temp_matrix*pca
	#Removing elements below the threshold from the positive layered matrix
	positive_bool_matrix = lil_matrix(layered_matrix > THRESHOLD)
	negative_bool_matrix = lil_matrix(layered_matrix < -THRESHOLD)
	#print np.sum(positive_bool_matrix)
	return positive_bool_matrix, negative_bool_matrix

#Add function to compare two layered networks and output a similarity score, based on this decide a threshold.
def get_layered_network_similarity(X, Y):
	M, NUM_SEQUENCES, LENGTH_SEQUENCES, TOTAL_TRAINING_SEQUENCES,TRAINING_SEQ, SCOPE, THRESHOLD, N = get_global_vars()
	score = X.multiply(Y)
	den =(X.sum(axis=1) +Y.sum(axis=1))/2
	#print "zeros in denominator", lil_matrix.sum(den==0), den.shape
	
	return np.mean(np.divide((score.sum(axis=1))[np.where(den!= 0)], den[np.where(den!= 0)]))

#Add function to return the R largest values in each row of the pca matrix
def find_largest_pca(orig_pca, R):
	y = []
	x = []
	data = []
	for i in range(1):
		max = orig_pca.max(axis=1)
		print max
	return orig_pca

#To read through all the files for the 50 users and generate co-occurence matrices for training sequences for the 50 users
def get_pca_decompositions():
	M, NUM_SEQUENCES, LENGTH_SEQUENCES, TOTAL_TRAINING_SEQUENCES,TRAINING_SEQ, SCOPE, THRESHOLD, N = get_global_vars()
    
	#Data matrix for storing training sequences in row major form
	training_data =csr_matrix((TOTAL_TRAINING_SEQUENCES, M*M), dtype = float)

	m=0
	command_to_index = {}
	command_to_occ = {}
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
		    y, data = generate_sequence_vector(command_list, command_to_index)
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
		np.savetxt("feature%d.csv" %(i+1), features_1_to_50[(TRAINING_SEQ*i):(TRAINING_SEQ*(i+1)),:], delimiter=",")
	w = csv.writer(open("dict.csv", "w"))
	for key, val in command_to_index.items():
    		w.writerow([key, val])
 
#Code to convert all these feature vectors into layered networks (N layered networks per sequence) and using them to find outputs for sequences 
#from the test set
def find_malicious_users():
	M, NUM_SEQUENCES, LENGTH_SEQUENCES, TOTAL_TRAINING_SEQUENCES,TRAINING_SEQ, SCOPE, THRESHOLD, N = get_global_vars()
        command_to_index = {}
	for key, val in csv.reader(open("dict.csv")):
    		command_to_index[key] = int(val)
		if int(val)>100000:
			print key, val
	pca = np.loadtxt(open("pca.csv","rb"),delimiter=",")
	mean_data = np.loadtxt(open("mean.csv","rb"),delimiter=",")
	R = 50
	pca = find_largest_pca(pca, R)
	
	test_data_malicious = np.zeros(shape = (100, 50))
	
	#To read through all the files for the 50 users and generate co-occurence matrices for training sequences for the 50 users
	#for i in [0]:
	for i in range(50):
	    print 'User%d' %(i+1)
	    training_features = np.loadtxt(open("feature%d.csv" %(i+1), "rb"), delimiter = "," )
	    reference_layered_network_pos = []
	    reference_layered_network_neg = []
            for j in range(TRAINING_SEQ):
		    temp1, temp2 =  get_layered_network(training_features[j, :], pca)
		    reference_layered_network_pos.append(temp1.astype(int))
		    reference_layered_network_neg.append(temp2.astype(int))
		    userNo = i+1
		    #print "Generating network for sequence %d" %j
	    file = open("User%d" % userNo, 'r')

	    counter = 0
	    #List to divide User data into sequences
	    command_list = []
	    seq_no = 0
		
	    for line in file:
		cmd = line.strip('\n')
		counter = counter+1
		#if counter <= 6600:
		if counter <= 5000:
		    continue
		command_list.append(cmd)
		#if counter > 6800:
		#	break
		if counter%LENGTH_SEQUENCES == 0:
		    y, data = generate_sequence_vector(command_list, command_to_index)
		    command_list = []
		    x = [0]*len(y)
		    seq_no = seq_no + 1
		    #print len(data)
		    #print max(data)
		    #print max(y)
		    sequence_matrix = csr_matrix((data, (x, y)),shape = (1, M*M)) - mean_data
		    features = sequence_matrix*pca.T
		    #print features.T.shape
		    sequence_layered_network_pos, sequence_layered_network_neg = get_layered_network(features.T, pca)
		    max_similarity = 0.0
		    #print 'Got layered network'
		    for j in range(TRAINING_SEQ):
                        #print 'computing similarity of user %d test seq %d with %d' %((i+1),seq_no, j)
                        sim = (get_layered_network_similarity(sequence_layered_network_pos, reference_layered_network_pos[j]) + get_layered_network_similarity(sequence_layered_network_neg, reference_layered_network_neg[j]))/2
                        if max_similarity < sim:
                            max_similarity = sim
                        #print sim
                    test_data_malicious[seq_no-1, i] = max_similarity
		    print "Test Sequence %d :%f\n" %(seq_no, max_similarity)
	np.savetxt("THRESHOLD_05/results.csv", test_data_malicious, delimiter=",")

def find_best_threshold():
	M, NUM_SEQUENCES, LENGTH_SEQUENCES, TOTAL_TRAINING_SEQUENCES,TRAINING_SEQ, SCOPE, THRESHOLD, N = get_global_vars()
	m2 = np.matrix(np.loadtxt(open("THRESHOLD_05/results.csv", 'rb'), delimiter = ','))
	m1 = np.matrix(np.loadtxt(open("reference.txt", 'rb'), delimiter = ' '))
	user_21_sim = m2[:,20]
	print m1[1, 20]
	m2 = np.delete(m2, 20, axis=1)
	m1 = np.delete(m1, 20, axis=1)
	malicious_users = np.sum(m1)
	#lis = list(xrange(100))
	lis = range(100)
	fp = []
	fn = []
	acc = []
	for thr in lis:
		thres = .01*thr
		temp = m2 < thres
		#print temp
		malicious_correctly_predicted = np.sum(np.multiply(m1, temp))
		#print true_correctly_predicted
		malicious_total_predicted = np.sum(temp)
		#print true_total_predicted
		fp.append((malicious_total_predicted - malicious_correctly_predicted)/(9000-malicious_users))
		fn.append((malicious_users - malicious_correctly_predicted)/(malicious_users))
		#acc.append(malicious_correctly_predicted/malicious_users)
	with open("THRESHOLD_05/threshold_finding.txt", "w") as myfile:
		myfile.write("False Positives")
		for item in fp:
			myfile.write("%f," % item)
		myfile.write("\n")
		myfile.write("False Negatives")
		for iter in fn:
			myfile.write("%f," % iter )
		myfile.write("\n")	
		
find_malicious_users()
find_best_threshold()
