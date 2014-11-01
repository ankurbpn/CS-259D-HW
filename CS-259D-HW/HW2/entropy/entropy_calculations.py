import matplotlib.pyplot as plt
import math
from datetime import datetime

def read_data_from_file():
	data = []
	f = open('logs.txt', 'r')
	for line in f:
		data.append(line.split())
	return data

def get_entropy_from_dict(dic, total):
	H = 0
	for key, val in dic.iteritems():
		H -= (1.0*val/total)*math.log(1.0*val/total)
	return H

def get_all_entropy_lists():
	lis_features = [3, 4, 5, 6, 7, 8]
	#lis_features = [3]
	pairwise_ent = {}
	ent = {}
	pairwise_count = {}
	count = {}
	dat = read_data_from_file()
	tim = []
	total = 0
	for item in dat:
		tim.append(datetime.strptime(item[2], '%H:%M:%S'))
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
		print tim[total-1]
	return tim, ent, pairwise_ent

def get_feature_name():
	dict = {}
	dict[3] = 'Duration'
	dict[4] = 'Serv'
	dict[5] = 'Src port'
	dict[6] = 'Dest Port'
	dict[7] = 'Src IP'
	dict [8] = 'Dest IP'
	return dict

def plot_all_entropy_lists():
	time, ent, pairwise_ent = get_all_entropy_lists()
	feat = get_feature_name()
	for i in ent.keys():
		plt.xlabel('Time')
		plt.ylabel('entropy ' + feat[i] )
		plt.plot(time, ent[i])
		plt.show()
	for i in pairwise_ent.keys():
		plt.xlabel('Time')
		plt.ylabel('entropy ' + feat[i[0]] + ' + ' + feat[i[1]])
		plt.plot(time, pairwise_ent[i])
		plt.show()


				
plot_all_entropy_lists()
#read_data_from_file()
