import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.model_selection import KFold
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import warnings
import sys
#from sklearn.utils.extmath import np.dot

warnings.simplefilter("error")

users = 0
items = 0

def readingFile(filename):
	f = open(filename,"r").readlines()
	data = []
	for row in f:
		r = row.split('\t')
		e = [int(r[2]),int(r[0].replace('nips',''))+1, int(r[3].replace('\n',''))]
		data.append(e)
	return data


def getData(filename):
	f = open(filename,"r")
	data = []
	for row in f:
		r = row.split(',')
		g = r[len(r)-1].split()
		data.append(g)
	data = np.array(data)
	data = data.astype('float')
	return data



def similarity(data,number):
	user_similarity_cosine = np.zeros((number,number))
	user_similarity_jaccard = np.zeros((number,number))
	user_similarity_pearson = np.zeros((number,number))
	for user1 in range(number):
		#print user1
		for user2 in range(number):
			if np.count_nonzero(data[user1]) and np.count_nonzero(data[user2]):
				user_similarity_cosine[user1][user2] = 1-scipy.spatial.distance.cosine(data[user1],data[user2])
				user_similarity_jaccard[user1][user2] = 1-scipy.spatial.distance.jaccard(data[user1],data[user2])
				try:
					if not math.isnan(scipy.stats.pearsonr(data[user1],data[user2])[0]):
						user_similarity_pearson[user1][user2] = scipy.stats.pearsonr(data[user1],data[user2])[0]
					else:
						user_similarity_pearson[user1][user2] = 0
				except:
					user_similarity_pearson[user1][user2] = 0



	return user_similarity_cosine, user_similarity_jaccard, user_similarity_pearson

def crossValidation(data, user_data, item_data):
	k_fold = KFold(n=len(data), n_folds=10)

	sim_user_cosine, sim_user_jaccard, sim_user_pearson = similarity(user_data,user_data.shape[0])
	sim_item_cosine, sim_item_jaccard, sim_item_pearson = similarity(item_data,item_data.shape[0])
	#sim_user_cosine, sim_user_jaccard, sim_user_pearson = np.random.rand(users,users), np.random.rand(users,users), np.random.rand(users,users)
	#sim_item_cosine, sim_item_jaccard, sim_item_pearson = np.random.rand(items,items), np.random.rand(items,items), np.random.rand(items,items) 

	'''sim_user_cosine = np.zeros((users,users))
	sim_user_jaccard = np.zeros((users,users))
	sim_user_pearson = np.zeros((users,users))

	f_sim = open("sim_user_hybrid.txt", "r")
	for row in f_sim:
		#print row
		r = row.strip().split(',')
		sim_user_cosine[int(r[0])][int(r[1])] = float(r[2])
		sim_user_jaccard[int(r[0])][int(r[1])] = float(r[3])
		sim_user_pearson[int(r[0])][int(r[1])] = float(r[4])
	f_sim.close()


	sim_item_cosine = np.zeros((items,items))
	sim_item_jaccard = np.zeros((items,items))
	sim_item_pearson = np.zeros((items,items))

	f_sim_i = open("sim_item_hybrid.txt", "r")
	for row in f_sim_i:
		#print row
		r = row.strip().split(',')
		sim_item_cosine[int(r[0])][int(r[1])] = float(r[2])
		sim_item_jaccard[int(r[0])][int(r[1])] = float(r[3])
		sim_item_pearson[int(r[0])][int(r[1])] = float(r[4])
	f_sim_i.close()'''

	rmse_cosine = []
	rmse_jaccard = []
	rmse_pearson = []

	for train_indices, test_indices in k_fold:
		train = [data[i] for i in train_indices]
		test = [data[i] for i in test_indices]

		M = np.zeros((users,items))

		for e in train:
			M[e[0]-1][e[1]-1] = e[2]

		true_rate = []
		pred_rate_cosine = []
		pred_rate_jaccard = []
		pred_rate_pearson = []

		for e in test:
			user = e[0]
			item = e[1]
			true_rate.append(e[2])

			user_pred_cosine = 1.0
			item_pred_cosine = 1.0

			user_pred_jaccard = 1.0
			item_pred_jaccard = 1.0

			user_pred_pearson = 1.0
			item_pred_pearson = 1.0

			#item-based
			if np.count_nonzero(M[:,item-1]):
				sim_cosine = sim_item_cosine[item-1]
				sim_jaccard = sim_item_jaccard[item-1]
				sim_pearson = sim_item_pearson[item-1]
				ind = (M[user-1] > 0)
				#ind[item-1] = False
				normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
				normal_jaccard = np.sum(np.absolute(sim_jaccard[ind]))
				normal_pearson = np.sum(np.absolute(sim_pearson[ind]))
				if normal_cosine > 0:
					item_pred_cosine = np.dot(sim_cosine,M[user-1])/normal_cosine

				if normal_jaccard > 0:
					item_pred_jaccard = np.dot(sim_jaccard,M[user-1])/normal_jaccard

				if normal_pearson > 0:
					item_pred_pearson = np.dot(sim_pearson,M[user-1])/normal_pearson


			#user-based
			if np.count_nonzero(M[user-1]):
				sim_cosine = sim_user_cosine[user-1]
				sim_jaccard = sim_user_jaccard[user-1]
				sim_pearson = sim_user_pearson[user-1]
				ind = (M[:,item-1] > 0)
				#ind[user-1] = False
				normal_cosine = np.sum(np.absolute(sim_cosine[ind]))
				normal_jaccard = np.sum(np.absolute(sim_jaccard[ind]))
				normal_pearson = np.sum(np.absolute(sim_pearson[ind]))
				if normal_cosine > 0:
					user_pred_cosine = np.dot(sim_cosine,M[:,item-1])/normal_cosine

				if normal_jaccard > 0:
					user_pred_jaccard = np.dot(sim_jaccard,M[:,item-1])/normal_jaccard

				if normal_pearson > 0:
					user_pred_pearson = np.dot(sim_pearson,M[:,item-1])/normal_pearson


			pred_cosine = (user_pred_cosine + item_pred_cosine)/2
			pred_jaccard = (user_pred_jaccard + item_pred_jaccard)/2
			pred_pearson = (user_pred_pearson + item_pred_pearson)/2
			pred_rate_cosine.append(pred_cosine)
			pred_rate_jaccard.append(pred_jaccard)
			pred_rate_pearson.append(pred_pearson)

		#print len(true_rate)
		#print len(pred_rate_cosine)
		rmse_cosine.append(sqrt(mean_squared_error(true_rate, pred_rate_cosine)))
		rmse_jaccard.append(sqrt(mean_squared_error(true_rate, pred_rate_jaccard)))
		rmse_pearson.append(sqrt(mean_squared_error(true_rate, pred_rate_pearson)))

		#print str(sqrt(mean_squared_error(true_rate, pred_rate_cosine))) + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_jaccard))) + "\t" + str(sqrt(mean_squared_error(true_rate, pred_rate_pearson)))
		#raw_input()

	#print sum(rms) / float(len(rms))
	rmse_cosine = sum(rmse_cosine) / float(len(rmse_cosine))
	rmse_pearson = sum(rmse_pearson) / float(len(rmse_pearson))
	rmse_jaccard = sum(rmse_jaccard) / float(len(rmse_jaccard))

	#print str(rmse_cosine) +	 "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson)

	f_rmse = open("rmse_hybrid.txt","w")
	f_rmse.write(str(rmse_cosine) + "\t" + str(rmse_jaccard) + "\t" + str(rmse_pearson) + "\n")

	rmse = [rmse_cosine, rmse_jaccard, rmse_pearson]
	req_sim = rmse.index(min(rmse))


	f_rmse.write(str(req_sim))
	f_rmse.close()

	if req_sim == 0:
		sim_mat_user = sim_user_cosine
		sim_mat_item = sim_item_cosine

	if req_sim == 1:
		sim_mat_user = sim_user_jaccard
		sim_mat_item = sim_item_jaccard

	if req_sim == 2:
		sim_mat_user = sim_user_pearson
		sim_mat_item = sim_item_pearson

	#predictRating(data, sim_mat_user, sim_mat_item)
	return sim_mat_user, sim_mat_item


def predictRating(data, user_data, item_data):

	sim_user, sim_item = crossValidation(data, user_data, item_data)

	M = np.zeros((users,items))
	for e in data:
		M[e[0]-1][e[1]-1] = e[2]

	pred_rate = np.zeros((users,items))

	#fw = open('result3.csv','w')
	fw_w = open('result3.csv','w')

#	l = len(toBeRated["user"])
	for e in range(users*items):
		user = (e/items) + 1
		item = (e%items) + 1

		user_pred = 1.0
		item_pred = 1.0

		#item-based
		if np.count_nonzero(M[:,item-1]):
			sim = sim_item[item-1]
			ind = (M[user-1] > 0)
			#ind[item-1] = False
			normal = np.sum(np.absolute(sim[ind]))
			if normal > 0:
				item_pred = np.dot(sim,M[user-1])/normal

		#user-based
		if np.count_nonzero(M[user-1]):
			sim = sim_user[user-1]
			ind = (M[:,item-1] > 0)
			#ind[user-1] = False
			normal = np.sum(np.absolute(sim[ind]))
			if normal > 0:
				user_pred = np.dot(sim,M[:,item-1])/normal


		pred_rate[user-1,item-1] = (user_pred + item_pred)/2
	np.save('predictions.npy',pred_rate)
	return pred_rate


#recommend_data = readingFile("ratings.csv")
recommend_data = readingFile(sys.argv[1])
user_data = getData(sys.argv[2])
item_data = getData(sys.argv[3])
users = user_data.shape[0]
items = item_data.shape[0]
#print users
#print items
predictRating(recommend_data, user_data, item_data)
#crossValidation(recommend_data, user_data, item_data)
# This file computes the complete matrix after collaborative filtering. Execute it as 'python cf1.py ratings.txt users.txt papers.txt' where users is the file containing the vector for users(one user per line), papers.txt is the file containing 
# the paper vectors to be predicted(one per line), and ratings.txt is  the file with relevance scores as given in 
# nips_reviewer_data.



#Hybrid Collaborative filtering done by giving similar items and similar users similar scores.
# (Metrics used for similarity: jaccard, cosine, pearson)
