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
	data = []
	int_to_vocab = {}
	vocab_to_int = {}
	return data, count, int_to_vocab, vocab_to_int 

# Our vocab size
vocab_size = 100000

if __name__ == '__main__':
		sentences, all_text =  read_data('data/HarryPotter.txt')	
		all_words = all_text.split()
		print ('Total Sentences:{}, Total characters:{}, Total Words:{}'.format(len(sentences), len(all_text), len(all_words)))
		#print (all_text[:150].split())
		data, count, int_to_vocab, vocab_to_int = build_vocab(all_words, vocab_size)
		print (data, count[:10], len(count)) 











