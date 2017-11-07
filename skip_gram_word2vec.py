# Importing all libraries
import tensorflow as tf
import numpy as np
import re
from collections import Counter
import random
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

def read_data(text_file):
	'''
	params: text_file
	Load and extract the text file. 
	Return the text
	'''
	text = ''
	data = open(text_file)
	data = data.read()
	text = ''.join(data)
	return text

def preprocess_data(text):
	'''
	Replace punctuation with tokens
	'''
	text = text.lower()
	text = text.replace('.', ' <PERIOD> ')
	text = text.replace(',', ' <COMMA> ')
	text = text.replace('"', ' <QUOTATION_MARK> ')
	text = text.replace(';', ' <SEMICOLON> ')
	text = text.replace('!', ' <EXCLAMATION_MARK> ')
	text = text.replace('?', ' <QUESTION_MARK> ')
	text = text.replace('(', ' <LEFT_PAREN> ')
	text = text.replace(')', ' <RIGHT_PAREN> ')
	text = text.replace('--', ' <HYPHENS> ')
	text = text.replace('?', ' <QUESTION_MARK> ')
	# text = text.replace('\n', ' <NEW_LINE> ')
	text = text.replace(':', ' <COLON> ')
	text = text.split()

	word_count = Counter(text)
	words = [word for word in text if word_count[word] > 5]

	return words


def prepare_data(words):
	'''
	Create lookup tables to map words to indices and viceversa
	'''
	count = Counter(words)
	sorted_count = sorted(count, key=count.get, reverse=True)
	vocab_to_int = {word:i for i, word in enumerate(sorted_count)} 
	int_to_vocab = {i:word for word, i in vocab_to_int.items()}

	return vocab_to_int, int_to_vocab

def sampling(data_to_ints, threshold=1e-5):
	'''
	Removing words such as 'the', 'of' etc.
	'''
	total = len(data_to_ints)
	index_count = Counter(data_to_ints)
	word_frequency = {word: count/total for word, count in index_count.items()}
	drop_prob = {word: 1-np.sqrt(threshold/word_frequency[word]) for word in index_count}
	final_words = [word for word in data_to_ints if random.random() < (1 - drop_prob[word])]

	return final_words

def window_words(words, index, window_size=5):
	'''
	returns a list of words in the window around the index
	'''
	s = np.random.randint(1, window_size+1)
	first = index - s if (index - s) > 0 else 0
	last = index + s
	close_words = set(words[first:index] + words[index+1:last+1])

	return list(close_words)

def generate_batches(words, batch_size, window_size=5):
	'''
	Returns batches(inputs and targets)
	'''
	batches = len(words) // batch_size
	words = words[:batch_size*batch_size]
	for i in range(0, len(words), batch_size):
		inputs, targets = [], []
		batch = words[i:i+batch_size]
		for ii in range(len(batch)):
			x = batch[ii]
			y = window_words(batch, ii, window_size)
			targets.extend(y)
			inputs.extend([x]*len(y))
		yield inputs, targets	

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


# Model Hyperparameters
batch_size = 512
embedding_size = 128
window_size = 5
sampled = 100
valid_size = 16
valid_window = 100
epochs = 10


if __name__ == '__main__':
	data = read_data('data/HarryPotter.txt')
	#print (data[:100])
	all_words = preprocess_data(data)
	#print (data[:100])
	vocab_to_int, int_to_vocab = prepare_data(all_words)
	vocab_size = len(int_to_vocab)
	data_to_ints = [vocab_to_int[word] for word in all_words]
	#print (data_to_ints[:10], all_words[:10])
	final_words = sampling(data_to_ints)


	## Build the network
	inputs = tf.placeholder(tf.int32, shape=(None))
	targets = tf.placeholder(tf.int32, shape=(None, None))


	embeddings = tf.Variable(tf.random_uniform((vocab_size, embedding_size), -1, 1))
	embed = tf.nn.embedding_lookup(embeddings, inputs)

	weights = tf.Variable(tf.truncated_normal((vocab_size, embedding_size)))
	biases = tf.Variable(tf.zeros(vocab_size))

	softmax_loss = tf.nn.sampled_softmax_loss(weights, biases, targets, embed, sampled, vocab_size)
	loss = tf.reduce_mean(softmax_loss)

	optimizer = tf.train.AdamOptimizer().minimize(loss)

	valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
	valid_examples = np.append(valid_examples, random.sample(range(1000, 1000+valid_window), valid_size//2))
	valid_set = tf.constant(valid_examples, dtype=tf.int32)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_set)
	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

	# Training the model
	init = tf.global_variables_initializer()
	session = tf.Session()
	saver = tf.train.Saver()
	
	with session as sess:
		sess.run(init)
		i = 1
		total_loss = 0
		for epoch in range(epochs+1):
			start_epoch = time.time()
			batches = generate_batches(final_words, batch_size, window_size)
			start = time.time()
			for x, y in batches:
				l, _ = sess.run([loss, optimizer], feed_dict={inputs:x, targets:np.array(y)[:, None]})
				total_loss += l

				if i % 100 == 0:
					end = time.time()
					print ("Epoch: {}/{}, Iteration: {}, Average loss: {:.4f}, Time/batch: {}".format(epoch, epochs, i, total_loss, (end-start)))

					total_loss = 0
					start = time.time()

				if i % 100 == 0:
					sim = similarity.eval()
					for j in range(valid_size):
						valid_word = int_to_vocab[valid_examples[j]]
						top = 8
						nearest = (-sim[j, :]).argsort()[1:top+1]		
						string = 'Nearest to {} : '.format(valid_word)	
						for k in range(top):
							closest = int_to_vocab[nearest[k]]
							string = string + " " + closest
						print (string)
				i += 1

			#end_epoch = time.time()
			#print ('Total Epoch Time : {:.4}'.format(end_epoch-start_epoch))

		saved_At = saver.save(sess, "checkpoints/final_100.ckpt")
		final_embeddings = sess.run(normalized_embeddings)					

		with open('embeddings.txt', 'wb') as f:
			print ("Saving Embeddings...")
			pickle.dump([final_embeddings, int_to_vocab], f, -1)


		# Plotting learned vector representations using TSNE
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
		new_embeddings = tsne.fit_transform(final_embeddings[:500, :])
		choosen_targets = [int_to_vocab[i] for i in range(500)]
		plot_embeddings(new_embeddings, choosen_targets)










