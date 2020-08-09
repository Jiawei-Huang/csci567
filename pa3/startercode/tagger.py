import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	state_dict = {value : key for key, value in enumerate(tags)}
	S = len(tags)
	sequence_init = np.zeros(S)
	transition = np.zeros((S, S))
	obs_dict = {}
	B = np.zeros((S, 1))

	for line in train_data:
		init_index = state_dict[line.tags[0]]
		sequence_init[init_index] += 1
		word = line.words[0]
		if len(obs_dict.keys()) == 0:
			obs_dict[word] = len(obs_dict.keys())
		elif word not in obs_dict.keys():
			obs_dict[word] = len(obs_dict.keys())
			B = np.concatenate((B, np.zeros((len(tags), 1), int)), axis=1)
		B[init_index, obs_dict[word]] += 1
		for i in range(len(line.tags) - 1):
			tag = line.tags[i]
			index = state_dict[tag]
			tag_next = line.tags[i+1]
			index_next = state_dict[tag_next]
			transition[index, index_next] += 1

			word = line.words[i+1]
			if word not in obs_dict.keys():
				obs_dict[word] = len(obs_dict.keys())
				B = np.concatenate((B, np.zeros((len(tags), 1), int)), axis=1)
			B[index_next, obs_dict[word]] += 1
	
	
	pi = sequence_init/sum(sequence_init)
	A = transition/np.sum(transition, axis=1).reshape(((len(tags), -1)))
	B = B/np.sum(B, axis=1).reshape((len(tags), -1))

	model = HMM(pi, A, B, obs_dict, state_dict)




	# S = len(tags)
	# # state_dict = {value : key for key, value in enumerate(tags)}
	# state_dict = {tags[i]: i for i in range(S)}

	# sequence_init = np.zeros(S)
	# transition = np.zeros((S, S))
	# obs_dict = {}
	# B = np.empty((S, 0))

	# for line in train_data:
	# 	init_index = state_dict[line.tags[0]]
	# 	sequence_init[init_index] += 1
	# 	word = line.words[0]

	# 	try:
	# 		B[init_index, obs_dict[word]] += 1
	# 	except:
	# 		obs_dict[word] = len(obs_dict)
	# 		B = np.concatenate((B, np.zeros((S, 1))), axis=1)
	# 		B[init_index, obs_dict[word]] += 1

	# 	# if len(obs_dict.keys()) == 0:
	# 	# 	obs_dict[word] = len(obs_dict.keys())
	# 	# elif word not in obs_dict.keys():
	# 	# 	obs_dict[word] = len(obs_dict.keys())
	# 	# 	B = np.concatenate((B, np.zeros((len(tags), 1))), axis=1)
	# 	# B[init_index, obs_dict[word]] += 1
	# 	for i in range(1, len(line.tags)):
	# 		tag = line.tags[i]
	# 		index = state_dict[tag]
	# 		tag_pre = line.tags[i-1]
	# 		index_pre = state_dict[tag_pre]
	# 		transition[index_pre, index] += 1

	# 		word = line.words[i]
	# 		try:
	# 			B[index, obs_dict[word]] += 1
	# 		except:
	# 			obs_dict[word] = len(obs_dict)
	# 			B = np.concatenate((B, np.zeros((S, 1))), axis=1)
	# 			B[index, obs_dict[word]] += 1

	# 		# if word not in obs_dict.keys():
	# 		# 	obs_dict[word] = len(obs_dict.keys())
	# 		# 	B = np.concatenate((B, np.zeros((len(tags), 1))), axis=1)
	# 		# B[index, obs_dict[word]] += 1
	
	
	# pi = sequence_init/np.sum(sequence_init)
	# A = transition/np.sum(transition, axis=1, keepdims=True)
	# B = B/np.sum(B, axis=1, keepdims=True)

	# model = HMM(pi, A, B, obs_dict, state_dict)

	# ----------------------------------------------------
	# S = len(tags)
	# state_dict = {tags[i]: i for i in range(S)}
	# obs_dict = {}
	
	# pi = np.zeros(S)
	# A = np.zeros((S, S))
	# B = np.zeros((S, 0))
	# for line in train_data:
	# 	pi[state_dict[line.tags[0]]] += 1

	# 	try:
	# 		idx = obs_dict[line.words[0]]
	# 	except:
	# 		idx = len(obs_dict)
	# 		obs_dict[line.words[0]] = idx
	# 		B = np.hstack((B, np.zeros((S, 1))))
	# 	B[state_dict[line.tags[0]]][idx] += 1

	# 	for i in range(1, line.length):
	# 		A[state_dict[line.tags[i-1]]][state_dict[line.tags[i]]] += 1

	# 		try:
	# 			idx = obs_dict[line.words[i]]
	# 		except:
	# 			idx = len(obs_dict)
	# 			obs_dict[line.words[i]] = idx
	# 			B = np.hstack((B, np.zeros((S, 1))))
	# 		B[state_dict[line.tags[i]]][idx] += 1

	# pi /= np.sum(pi)
	# A /= np.sum(A, axis=1, keepdims=True)
	# B /= np.sum(B, axis=1, keepdims=True)


	# model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	for line in test_data:
		tagging.append(model.viterbi(line.words))
	###################################################
	return tagging
