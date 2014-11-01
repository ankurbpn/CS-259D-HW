##from matplotlib import pyplot
import math

def read_data_from_file():
	data = []
	f = open('logs.txt', 'r')
	for line in f:
		data.append(line.split())
	return data

def get_entropy_from_dict(dic, total):
	H = 0
	for key, val in dic.iteritems():
		H -= (1.0*val/total)*math.log(val/total)
	return H

def get_all_entropy_lists():
	lis_features = [3, 4, 5, 6, 7, 8]
	pairwise_ent = {}
	ent = {}
	pairwise_count = {}
	count = {}
	dat = read_data_from_file()
	total = 0
	for item in dat:
		total += 1
		for i in lis_features:
			if i in count.keys():
				if item[i] in count[i].keys():
					count[i][item[i]]+=1
				else:
					count[i][item[i]]=1
			else:
				count[i] = {}
				count[i][item[i]]=1				

			if i in ent.keys():
				ent[i].append(get_entropy_from_dict(count[i], total))
			else:
				ent[i] = [get_entropy_from_dict(count[i], total)]

			for j in lis_features:
				if i!=j:
				if (i, j) in pairwise_count.keys():
					if (item[i], item[j]) in pairwise_count[(i, j)].keys():
						pairwise_count[(i, j)][(item[i], item[j])]+=1
					else:
						pairwise_count[(i, j)][(item[i], item[j])]=1
				else:
					pairwise_count[(i, j)] = {}
					pairwise_count[(i, j)][(item[i], item[j])]=1				

				if (i, j) in pairwise_ent.keys():
					pairwise_ent[(i, j)].append(get_entropy_from_dict(pairwise_count[(i, j)], total))
				else:
					pairwise_ent[(i, j)] = [get_entropy_from_dict(pairwise_count[(i, j)], total)]
		print total

				
get_all_entropy_lists()
#read_data_from_file()
