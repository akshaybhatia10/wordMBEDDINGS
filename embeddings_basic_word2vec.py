# Importing all libraries

import tensorflow as tf
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
import sys
import re
import collections
import random
import pickle
import argparse
from six.moves import xrange  



def read_data(text_file):
	'''
	params: text_file
	Load and extract the text file. 
	Return all the sentences in a list  and complete text
	'''
	all_text = ''
	data = open(text_file)
	data = data.read()
	all_text = ''.join(data)
	sentences = all_text.split('\n')
	return sentences, all_text

def build_vocab(words, vocab_size):
	'''
	params: words, vocab_size
	Process raw inputs and build a dictionary 
	and replace rare words with UNK token. 
	returns 
	'''
	count = [['UNK', -1]]
	count.extend(Counter(words).most_common(vocab_size))
	
	# Mapping words to indices:
	# Most frequent word comes first, followed by second and so on
	# {'the': 1, 'to': 2} ......
	vocab_to_int = {}	
	for word, i in count:
		vocab_to_int[word] = len(vocab_to_int)

	unknown_count = 0
	data = []
	for word in words:
		if word in vocab_to_int:
			idx = vocab_to_int[word]
		else:
			idx = 0
			unknown_count += 1
		data.append(idx)
	count[0][1] = unknown_count
	
	## Mapping from indices to words: reverse of vocab_to_int			
	int_to_vocab = {}
	for (word, index) in vocab_to_int.items():
		int_to_vocab[index] = word
	
	return data, count, int_to_vocab, vocab_to_int 

# Our vocab size
vocab_size = 50000
index = 0

def get_batches(batch_size, num_skips, window_size):
	'''
	params: batch_size, num_skips, window_size
	Generate batches 	
	'''
	global index
	inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
	targets = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

	# skip window size
	total_window_size = 2*window_size + 1
	cache = collections.deque(maxlen=total_window_size)
	if index + total_window_size > len(data):
		index = 0
	cache.extend(data[index:index + total_window_size])	

	index += total_window_size
	for i in range(batch_size // num_skips):
		t = window_size
		t_avoid = [window_size]
		for j in range(num_skips):
			while t in t_avoid:
				t = random.randint(0, total_window_size - 1)
			t_avoid.append(t)
			inputs[i*num_skips + j] = cache[window_size]
			targets[i*num_skips + j, 0] = cache[t]

		if index == len(data):
			cache[:] = data[:total_window_size]
			index = total_window_size
		else:
			cache.append(data[index])
			index += 1				

	index = (index + len(data) -  total_window_size) % len(data)		
	return inputs, targets


if __name__ == '__main__':

	sentences, all_text =  read_data('data/HarryPotter.txt')	
	all_words = all_text.split()
	print ('Total Sentences: {}, Total characters: {}, Total Words: {}'.format(len(sentences), len(all_text), len(all_words)))
	#print (all_text[:150].split())
	data, count, int_to_vocab, vocab_to_int = build_vocab(all_words, vocab_size)
	#print (data[:5], count[:5], len(count)) 
	#print (vocab_to_int['the'])
	#print (int_to_vocab[0])
	print (data[:10], [int_to_vocab[i] for i in data[:10]])
	inputs, targets = get_batches(batch_size=8, num_skips=4, window_size=2)

	for i in range(8):
		print (inputs[i], int_to_vocab[inputs[i]], ' is ', targets[i, 0], int_to_vocab[targets[i, 0]])

	# Model Hyperparameters
	batch_size = 128
	embedding_size = 128
	window_size = 2
	num_skips = 4	

	valid_size = 16
	valid_window = 100
	valid_example = np.random.choice(valid_window, valid_size, replace=False)
	num_sampled = 64

	# Building the word2vec model
	model_inputs = tf.placeholder(tf.int32, shape=(batch_size))
	model_targets = tf.placeholder(tf.int32, shape=(batch_size, 1))
	valid_set = tf.constant(valid_example, dtype=tf.int32)

	embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
	embed = tf.nn.embedding_lookup(embeddings, model_inputs)

	weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0/ math.sqrt(embedding_size)))
	bias = tf.Variable(tf.zeros([vocab_size]))

	nce_loss = tf.nn.nce_loss(weights=weights,
							  biases=bias,
							  labels=model_targets,
							  inputs=embed,
							  num_sampled=num_sampled,
							  num_classes=vocab_size)

	loss = tf.reduce_mean(nce_loss)

	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_set)
	similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

	init = tf.global_variables_initializer()
	session = tf.Session()

	epochs = 100

	with session as sess:
		init.run()

		total_loss = 0
		for epoch in range(epochs):
			i, t = get_batches(batch_size, num_skips, window_size)

			_, loss_value = sess.run([optimizer, loss], feed_dict={model_inputs:i, model_targets:t})
			total_loss += loss_value

			if epoch % 20 == 0:
				if epoch > 0:
					total_loss/=20
				print ("Average Loss at epoch ", epoch, ' = ', total_loss)
				total_loss = 0

			if epoch % 40 == 0:
				sim = similarity.eval()
				for j in range(valid_size):
					valid_word = int_to_vocab[valid_example[j]]
					top = 8
					nearest = (-sim[j, :]).argsort()[1:top+1]
					string = 'Nearest to {}'.format(valid_word)
					for k in range(top):
						closest = int_to_vocab[nearest[k]]
						string = string + closest
					print (string)	

		final_embeddings = normalized_embeddings.eval()		