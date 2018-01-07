import time
import numpy as np
import tensorflow as tf
import re
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

PATH = '../corpus/test.txt'

text = ''
with open(PATH, encoding='utf-8') as f:
    text = f.read()

text = re.sub("[^a-zA-Z]", " ", text)
words = text.lower().split()

del text;
vocab = Counter(words)
vocab = sorted(vocab, key=vocab.get, reverse=True)

vocab_to_int = {word:i for i, word in enumerate(vocab)} 
int_to_vocab = {i:word for word, i in vocab_to_int.items()}

vocab_size = len(vocab)
print ('Vocab Size', vocab_size)
window_size = 2
input_output_pairs = []

def generate_data():
    for i, word in enumerate(words):
        output = words[i+1:window_size+i+1] + words[i-window_size:i] if i > 0 else words[i+1:window_size+i+1]
        for w in output:
            input_output_pairs.append([word, w])

def convert_to_one_hot(i):
    oh_vector = np.zeros(vocab_size)
    oh_vector[i] = 1
    return oh_vector

def calculate_distance(embedding_one, embedding_two):
    return np.sqrt(np.sum((embedding_one - embedding_two)**2))

def find_closest(word, embeddings):
    minimum_distance = 10000
    minimum = -1
    idx = vocab_to_int[word]
    v = embeddings[idx]
    for i, embedding in enumerate(embeddings):
        if calculate_distance(embedding, v) < minimum_distance and not np.array_equal(v, embedding):
            minimum_distance = calculate_distance(embedding, v)
            minimum = i
    return int_to_vocab[minimum] 

def plot_embeddings(embeddings, targets,file='basic.jpg'):
    '''
    Plot the n-dimensional embeddings
    '''
    plt.figure(figsize=(18,18))
    for (i, target) in enumerate(targets):
        x, y = embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(target, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(file)    

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
print (X.shape, Y.shape)

batch_size = 64
embedding_size = 100
epochs = 10

inputs = tf.placeholder(tf.float32, shape=(None, vocab_size))
targets = tf.placeholder(tf.float32, shape=(None, vocab_size))

embedding_weights = tf.Variable(tf.truncated_normal(shape=(vocab_size, embedding_size)))
embedding_bias = tf.Variable(tf.zeros(shape=(embedding_size)))
h = tf.add(tf.matmul(inputs, embedding_weights), embedding_bias)

softmax_weights = tf.Variable(tf.truncated_normal(shape=(embedding_size, vocab_size)))
softmax_bias = tf.Variable(tf.zeros(shape=(vocab_size)))
output = tf.add(tf.matmul(h, softmax_weights), softmax_bias)
predictions = tf.nn.softmax(output)

#softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets,logits=predictions)
softmax_loss = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(predictions), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(softmax_loss)

init = tf.global_variables_initializer()
session = tf.Session()
saver = tf.train.Saver()

with session as sess:
    sess.run(init)
    start = time.time()
    for epoch in range(epochs):
        start = time.time()
        sess.run(optimizer, feed_dict={inputs:X, targets:Y})
        print ('Epoch: {}, Loss:{}, Time taken:{:.3f} seconds'.format(epoch, sess.run(softmax_loss, feed_dict={inputs:X, targets:Y}), (time.time() - start)))

    embeddings = sess.run(embedding_weights + embedding_bias)
    #print (embeddings[vocab_to_int['harry']])
    #print (embeddings[vocab_to_int['potter']])
    print (find_closest('harry', embeddings))   
    print (find_closest('triwizard', embeddings))
    print (find_closest('fudge', embeddings))
    print (find_closest('hogwarts', embeddings))

    # Plotting learned vector representations using TSNE
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    new_embeddings = tsne.fit_transform(embeddings[:200, :])
    choosen_targets = [int_to_vocab[i] for i in range(200)]
    plot_embeddings(new_embeddings, choosen_targets)
