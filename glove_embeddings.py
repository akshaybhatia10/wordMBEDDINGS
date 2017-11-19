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


if __name__ == '__main__':
	data = read_data('corpus/HarryPotter.txt')[:100]
	all_text, all_words = preprocess_data(data)
	print (all_text[:200])
