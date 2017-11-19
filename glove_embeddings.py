from collections import Counter, defaultdict
import os
import random
import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords


def read_data(text_file):
	'''
	params: text_file
	Load and extract the text file. 
	Return the text
	'''
	text = ''
	data = open(text_file, encoding="utf-8")
	data = data.read()
	text = ''.join(data)
	return text

def preprocess_data(text):
	'''
	clean dataset - replace numbers and stopwords with " " 
	'''	
	text = re.sub("[^a-zA-Z]", " ", text)
	words = text.lower().split()
	all_text = ' '.join(words)
	return all_text, words

def build_vocab(words, vocab_size):
	'''
	params: words, vocab_size
	Process raw inputs and build a dictionary 
	and replace rare words with UNK token (index 0). 
	returns 
	'''
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
	
	return data, count, int_to_vocab, vocab_to_int 

vocab_size = 50000

if __name__ == '__main__':
	corpus = read_data('corpus/HarryPotter.txt')
	text, words = preprocess_data(corpus)
	#print (words[:50])
	data, count, int_to_vocab, vocab_to_int = build_vocab(words, vocab_size)



