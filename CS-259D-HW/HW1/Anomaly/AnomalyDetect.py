from scipy.sparse import csr_matrix
#Global parameters
#Index numer for commands, will get updated as we encounter new commands
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

#Global map from command name to index number
commandToIndex = {}

#Data matrix for storing training sequences in row major form
trainingData = csr_matrix((TOTAL_TRAINING_SEQUENCES, M*M), dtype = float)

#Data vector for storing mean of the training sequences for centering
#meanData = csr_matrix((1, M*M), dtype = float)
#numVals = csr_matrix((1, M*M), dtype = float)

#Function to generate a co-occurence matrix vector of length M*M from command sequence
def generateSequenceVector(sequence):
	global commandToIndex
	global SCOPE

	#Currently adds one for every co-occurence, update to Gaussian?
	y = []
	data = []
	for i in range(len(sequence)):
		for j in range(SCOPE):
			if i + j + 1 < len(sequence):
				listIndex = commandToIndex.get(sequence[i])*M + commandToIndex.get(sequence[j + i + 1])
				if listIndex not in y:
					y.append(listIndex)
					data.append(1)
				else:
					ind = y.index(listIndex)
					data[ind] = data[ind] + 1
	return y, data




#To read through all the files for the 50 users and generate co-occurence matrices for training sequences for the 50 users
for i in range(50):
	userNo = i+1
	#print "User %d" % userNo
	file = open("User%d" % userNo, 'r')

	counter = 0
	#List to divide User data into sequences
	commandList = []
	seqNo = 0

	for line in file:
		counter = counter+1
		if counter > 5000:
			break
		commandList.append(line)
		if not commandToIndex.has_key(line):
			commandToIndex[line] = m
			#print m, line
			m = m+1

		if counter%LENGTH_SEQUENCES == 0:
			y, data = generateSequenceVector(commandList)
			commandList = []
			x = [seqNo + i*TRAINING_SEQ]*len(y)
			seqNo = seqNo + 1
			trainingData = trainingData + csr_matrix((data, (x, y)), shape = (TOTAL_TRAINING_SEQUENCES, M*M))
			#meanData = meanData + csr_matrix((data, ([0]*len(y), y)), shape = (1, M*M))
			#numVals = numVals + csr_matrix(([1]*len(y), ([0]*len(y), y)), shape = (1, M*M))


	if len(commandList) > 0:
		y, data = generateSequenceVector(commandList)
		commandList = []
		trainingData = trainingData + csr_matrix((data, (x, y)), shape = (TOTAL_TRAINING_SEQUENCES, M*M))
		#meanData = meanData + csr_matrix((data, ([0]*len(y), y)), shape = (1, M*M))
		#numVals = numVals + csr_matrix(([1]*len(y), ([0]*len(y), y)), shape = (1, M*M))
 
#Centering training data
meanData = trainingData.mean(axis = 1)
trainingDataCentered = trainingData - (meanData*([1]*TOTAL_TRAINING_SEQUENCES).T).T
del trainingData