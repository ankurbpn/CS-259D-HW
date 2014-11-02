#import matplotlib.pyplot as plt
import math
from datetime import datetime
import pickle
import multiprocessing

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
	pickle.dump((tim, ent, pairwise_ent), open('entropies.pickle',  'wb'))

def get_sliding_window_entropy_list():
	N = 100
	Nby2 = 50
	lis_features = [3, 4, 5, 6, 7, 8]
	#lis_features = [3]
	pairwise_ent = {}
	ent = {}
	pairwise_count = {}
	count = {}
	new_count = {}
	new_p_count = {}
	dat = read_data_from_file()
	tim = []
	total = 0
	for item in dat:
		#tim.append(datetime.strptime(item[2], '%H:%M:%S'))
		total += 1
		if total%Nby2==0:
			tim.append(datetime.strptime(item[2],  '%H:%M:%S'))

		for i in lis_features:
			if i in count.keys():
				if item[i] in count[i].keys():
					count[i][item[i]]+=1
				else:
					count[i][item[i]]=1
			else:
				count[i] = {}
				count[i][item[i]]=1

			if i in new_count.keys():
				if item[i] in new_count[i].keys():
					new_count[i][item[i]]+=1
				else:
					new_count[i][item[i]]=1
			else:
				new_count[i] = {}
				new_count[i][item[i]]=1				
			
			if total%Nby2==0:
							if i in ent.keys():
					ent[i].append(get_entropy_from_dict(count[i], N))
				else:
					ent[i] = [get_entropy_from_dict(count[i], N)]
				count[i] = new_count[i]
				new_count[i] = {}

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
					
					if (i, j) in new_p_count.keys():
						if (item[i], item[j]) in new_p_count[(i, j)].keys():
							new_p_count[(i, j)][(item[i], item[j])]+=1
						else:
							new_p_count[(i, j)][(item[i], item[j])]=1
					else:
						new_p_count[(i, j)] = {}
						new_p_count[(i, j)][(item[i], item[j])]=1				

					if total%Nby2==0:
		
						if (i, j) in pairwise_ent.keys():
							pairwise_ent[(i, j)].append(get_entropy_from_dict(pairwise_count[(i, j)], N))
						else:
							pairwise_ent[(i, j)] = [get_entropy_from_dict(pairwise_count[(i, j)], N)]
						pairwise_count[(i, j)] = new_p_count[(i, j)]
						new_p_count[(i, j)] = {}
		print item[2]
	pickle.dump((tim, ent, pairwise_ent), open('windowed_entropies.pickle',  'wb'))


def get_feature_name():
	dict = {}
	dict[3] = 'Duration'
	dict[4] = 'Serv'
	dict[5] = 'SrcPort'
	dict[6] = 'DestPort'
	dict[7] = 'SrcIP'
	dict [8] = 'DestIP'
	return dict

def get_triplet_entropy():
	trip = [4, 7, 8]
	ent = []
	count = {}
	dat = read_data_from_file()
	tim = []
	total = 0
	for item in dat:
		total+=1
		tim.append(datetime.strptime(item[2], '%H:%M:%S'))
		key = (item[trip[0]], item[trip[1]], item[trip[2]])
		if key in count.keys():
			count[key]+=1
		else:
			count[key]=1
		ent.append(get_entropy_from_dict(count, total))	
		print tim[total-1]	
	pickle.dump((tim, ent), open('trip_entropy.pickle',  'wb'))


def plot_all_entropy_lists(Triplet = False):
	if Triplet:
		time, ent = pickle.load(open('trip_entropy.pickle',  'rb'))
		#ent = normalize_ent_for_comparison(ent)
		feat = get_feature_name()	
		fig = plt.figure()
	#	count += 1
		plt.xlabel('Time')
		plt.ylabel('entropy ' + feat[4] + ' + ' + feat[7] + ' + ' + feat[8])
		plt.plot(time, ent)
		plt.show()
		fig.savefig('triplet.png')

	else:
		count = 0
		time, ent, pairwise_ent = pickle.load(open('windowed_entropies.pickle', 'rb'))
		feat = get_feature_name()
		for i in ent.keys():
			fig = plt.figure(count)
		#	ent[i] = normalize_ent_for_comparison(ent[i])
			count+=1
			plt.xlabel('Time')
			plt.ylabel('entropy ' + feat[i] )
			print len(time), len(ent[i])
			plt.plot(time, ent[i])
			#plt.show()
			fig.savefig('w_'+feat[i] + '.png')
		for i in pairwise_ent.keys():
			fig = plt.figure(count)
			count += 1
			plt.xlabel('Time')
			plt.ylabel('entropy ' + feat[i[0]] + ' + ' + feat[i[1]])
		#	pairwise_ent[i] = normalize_ent_for_comparison(pairwise_ent[i])
			plt.plot(time, pairwise_ent[i])
			#plt.show()
			fig.savefig('w_'+feat[i[0]] + '_' + feat[i[1]] + '.png')

def normalize_ent_for_comparison(ent):
	for i in range(len(ent)):
		if i>1:
			ent[i] = ent[i]/math.log(i)
	return ent

get_sliding_window_entropy_list()
#get_triplet_entropy()
#get_all_entropy_lists()
#plot_all_entropy_lists()
#read_data_from_file()

