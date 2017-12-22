"""
Generate word embeddings using Glove algorithm
"""
from collections import Counter, defaultdict
import collections
import os
import random
import tensorflow as tf
import nltk
import re
import pickle
from nltk.corpus import stopwords
from pprint import pprint as pp
from scipy.sparse import lil_matrix
import numpy as np
import time


def read_data(path):

	"""Loads and Extracts text from given file.

    Args:
        path (str): Path to text file.

    Returns:
        Complete text of the file including special characters.
	
	"""
	text = ''
	with open(path,  'r', encoding="utf-8") as f:
		data = f.readlines()
		text = ''.join(data)
	
	return text

def preprocess_data(text):
	"""Preprocesses text and removes special character including numbers.

    Args:
        text (str): Text to be preprocessed.

    Returns:
        Tuple of complete preprocessed text and words.
	
	"""
	text = re.sub("[^a-zA-Z]", " ", text)
	words = text.lower().split()
	all_text = ' '.join(words)
	return (all_text, words)

def build_vocab(words, vocab_size):
	"""Process raw inputs and build a dictionary and replace rare words with UNK token (index 0). 
	
	Args:
		words (list): The complete word list
		vocab_size (int): size of the vocabulary
	
	Returns: 
		Tuple of 4 objects namely data, count, int_to_vocab, vocab_to_int
	"""
	count = [['UNK', -1]]
	count.extend(Counter(words).most_common(vocab_size - 1))
	
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
	#for (word, index) in vocab_to_int.items():
	int_to_vocab = dict(zip(vocab_to_int.values(), vocab_to_int.keys()))
	
	return (data, count, int_to_vocab, vocab_to_int) 


def get_batches(batch_size, num_skips, window_size):
	"""Generate batches of inputs(context), targets and weights( denote how far the context word is from the target word)	
	
	Args:
		batch_size (int): The size of each batch 
		num_skips (int): Number of times to reuse an input word for a target word
		window_size (int): Number of words to consider left and right.

	Returns:
		batches of input words, target words and corresponding weights
	"""
	global index

	inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
	targets = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	weights = np.ndarray(shape=(batch_size), dtype=np.float32)

	# skip window size
	total_window_size = 2*window_size + 1
	cache = collections.deque(maxlen=total_window_size)
	
	for _ in range(total_window_size):
		cache.append(data[index])
		index = (index + 1) % len(data)

	for i in range(batch_size // num_skips):
		t = window_size
		t_avoid = [window_size]
		for j in range(num_skips):
			while t in t_avoid:
				t = random.randint(0, total_window_size - 1)
			t_avoid.append(t)
			inputs[i*num_skips + j] = cache[window_size]
			targets[i*num_skips + j, 0] = cache[t]
			weights[i*num_skips + j] = abs(1.0/(t - window_size))

	cache.append(data[index])
	index = (index + 1) % len(data)		
	
	return inputs, targets, weights


def get_cooccurrence_matrix(batch_size, num_skips, window_size):
	"""Generate Co occurrence matrix	
	
	Args:
		batch_size (int): The size of each batch 
		num_skips (int): Number of times to reuse an input word for a target word
		window_size (int): Number of words to consider left and right.
	"""
	index = 0
	tot_iterations = data_count//batch_size
	print('Total Iterations: {}'.format(tot_iterations))
	for i in range(tot_iterations):
		inputs, targets, weights = get_batches(batch_size, num_skips, window_size)
		targets = targets.reshape(-1)

		for inp, target, weight in zip(inputs, targets, weights):
			cooccurrence_matrix[inp, target] += (1.0*weight)

def plot_embeddings(embeddings, targets, file='vis_word2vec.png'):
	'''
	Plot the n-dimensional embeddings
	'''
	plt.figure(figsize=(18,18))
	for (i, target) in enumerate(targets):
		x, y = embeddings[i, :]
		plt.scatter(x, y)
		plt.annotate(target, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
	plt.savefig(file)			


if __name__ == '__main__':

	vocab_size = 50000
	index = 0
	batch_size = 128
	embedding_size = 128
	num_skips = 8
	window_size = 4
	mat_idx = 0

	valid_size = 16
	valid_window = 100

	epochs = 5000

	corpus = read_data('../corpus/HarryPotter.txt')
	text, words = preprocess_data(corpus)
	#print(text[:200], '\n', words[:50])
	data, count, int_to_vocab, vocab_to_int = build_vocab(words, vocab_size)
	data_count = len(data)
	#print(data[:100])
	#for word in words[:50]:
	#	print ('Word: {} , index: {}'.format(word, vocab_to_int[word]))

	cooccurrence_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.float32) 
	print (cooccurrence_matrix.shape)
	get_cooccurrence_matrix(batch_size=8, num_skips =num_skips, window_size=window_size)	

	inputs = tf.placeholder(tf.int32, shape=(batch_size))
	targets = tf.placeholder(tf.int32, shape=(batch_size, batch_size))

	embeddings = tf.Variable(tf.random_uniform((vocab_size, embedding_size), -1, 1, dtype=tf.float32))
	b_embeddings = tf.Variable(tf.random_uniform((vocab_size), 0.0, 0.01, dtype=tf.float32))

	in_embed = tf.nn.embedding_lookup(embeddings, inputs)
	out_embed = tf.nn.embedding_lookup(embeddings, targets)
	b_in_embed = tf.nn.embedding_lookup(b_embeddings, inputs)
	b_out_embed = tf.nn.embedding_lookup(b_embeddings, targets)

	w1 = tf.placeholder(tf.float32, shape=(batch_size))
	w2 = tf.placeholder(tf.float32, shape=(batch_size))

	glove_loss = w1 * (tf.reduce_sum(in_embed*out_embed, axis=1) + b_in_embed + b_out_embed - tf.log(1+w2))**2
	loss = tf.reduce_mean(glove_loss)
	optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

	valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
	valid_examples = np.append(valid_examples, random.sample(range(1000 + 1000+valid_window), valid_size//2))
	valid_set = tf.constant(valid_examples, dtype=tf.int32)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.lookup(normalized_embeddings, valid_set)
	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

	# Training 

	init = tf.global_variables_initializer()
	sess= tf.Session()
	saver = tf.Saver()

	with sess:
		sess.run(init)
		i = 0
		total_loss = 0
		for epoch in range(epochs):
			start_epoch = time.time()
			batches = get_batches(batch_size, num_skips, window_size)
			batch_w1 = []
			batch_w2 = []
			start = time.time()
			for x, y, _ in zip(batches[0], batches[1].reshape(-1), batches[2]):
				batch_w1.append((np.asscalar(cooccurrence_matrix[x, y])/ 100.0)**0.75)
				batch_w2.append(cooccurrence_matrix[x, y])

			batch_w1 = np.clip(batch_w1, -100, 1)
			batch_w2 = np.asarray(batch_w2)	

			feed_dict = {inputs: batches[0].reshape(-1), targets: batches[1].reshape(-1),
						w1: batch_w1, w2:batch_w2}
			l, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
			total_loss += l

			if i % 2000 == 0:
				end = time.time()
				total_loss = total_loss / 2000 
				print ("Epoch: {}/{}, Iteration: {}, Average loss: {:.4f}, Time/batch: {}".format(epoch, epochs, i, total_loss, (end-start)))
				total_loss = 0
				start = time.time()

			if i % 5000 == 0:
				sim = similarity.eval()
				for j in range(valid_size):
					valid_word = int_to_vocab[valid_examples[j]]
					top = 8
					nearest = (-sim[j : ]).argsort()[1: top+1]
					string = 'Nearest to {} : '.format(valid_word)
					for k in range(top):
						closest = int_to_vocab[nearest[k]]
						string += " " + closest
					print (string)		
			i += 1

		saved_At = saver.save(sess, "checkpoints/final_10000.ckpt")
		final_embeddings = sess.run(normalized_embeddings)

		with open('embeddings.txt', 'wb') as f:
			print ("Saving Embeddings...")
			pickle.dump([final_embeddings, int_to_vocab], f, -1)			

		
		# Plotting learned vector representations using TSNE
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
		new_embeddings = tsne.fit_transform(final_embeddings[:500, :])
		choosen_targets = [int_to_vocab[i] for i in range(500)]
		plot_embeddings(new_embeddings, choosen_targets)
	
