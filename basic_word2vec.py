import time
import numpy as np
import tensorflow as tf
import re
from collections import Counter

PATH = '../corpus/HarryPotter.txt'

text = ''
with open(PATH, encoding='utf-8') as f:
    text = f.read()

text = re.sub("[^a-zA-Z]", " ", text)
words = text.lower().split()

del text;
vocab = Counter(words)
vocab = sorted(vocab, key=vocab.get, reverse=True)[:5000]

vocab_to_int = {word:i for i, word in enumerate(vocab)} 
int_to_vocab = {i:word for word, i in vocab_to_int.items()}

vocab_size = len(vocab)
window_size = 2
input_output_pairs = []

def generate_data():
    for i, word in enumerate(words):
        if word not in vocab:
            output = 
        output = words[i+1:window_size+i+1] + words[i-window_size:i] if i > 0 else words[i+1:window_size+i+1]
        for w in output:
            input_output_pairs.append([word, w])

def convert_to_one_hot(i):
    oh_vector = np.zeros(vocab_size)
    oh_vector[i] = 1
    return oh_vector

generate_data()

print ('Seperating Data')
start = time.time()
x, y = [], []
for i in range(len(input_output_pairs)):
    ip = input_output_pairs[i][0]
    op = input_output_pairs[i][1]
    x.append(ip)
    y.append(op)
end = time.time()
print ('Seperating Data Done: ', end-start)

del input_output_pairs; del vocab; del words;

print ('Converting Data')
start = time.time()
X, Y = [], []
for i,t in zip(x, y):
    X.append(convert_to_one_hot(vocab_to_int[i]))
    Y.append(convert_to_one_hot(vocab_to_int[t]))
end = time.time()
print ('Converting Data Done: ', end-start)

X, Y = np.asarray(X), np.asarray(Y)
print (x.shape, y.shape)

batch_size = 32
embedding_size = 50
epochs = 100

inputs = tf.placeholder(tf.float32, shape=(None, vocab_size))
targets = tf.placeholder(tf.float32, shape=(None, vocab_size))

embedding_weights = tf.Variable(tf.truncated_normal(shape=(vocab_size, embedding_size)))
embedding_bias = tf.Variable(tf.zeros(shape=(embedding_size)))
h = tf.add(tf.matmul(inputs, embedding_weights), embedding_bias)

softmax_weights = tf.Variable(tf.truncated_normal(shape=(embedding_size, vocab_size)))
softmax_bias = tf.Variable(tf.zeros(shape=(vocab_size)))
pred = tf.nn.softmax(tf.add(tf.matmul(h, embedding_bias), softmax_bias))

softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets,logits=predictions)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(softmax_loss)

init = tf.global_variables_initializer()
session = tf.Session()
saver = tf.train.Saver()

with session as sess:
    sess.run(init)
    for epoch in range(epoch):
        sess.run(optimizer, feed_dict={inputs:X, targets:Y})
        print ('Epoch: {}, Loss:{}'.format(epoch, sess.run(softmax_loss, feed_dict={inputs:X, targets:Y})))




