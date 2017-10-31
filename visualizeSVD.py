import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Some dummy sentances
sentences = ["The Prime Minister could not honestly return this compliment, so said nothing at all",
			 "Naturally, he had thought that the long campaign and the strain of the election had caused him to go mad",
			 "It was nearing midnight and the Prime Minister was sitting alone in his office, reading a long memo that was slipping through his brain without leaving the slightest trace of meaning behind",
			 "A team of Healers from St. Mungo's Hospital for Magical Maladies and Injuries are examining him as we speak",
			 "Horace, said Dumbledore, relieving Harry of the responsibility to say any of this, likes his comfort. He also likes the company of the famous, the successful, and the powerful. He enjoys the feeling that he influences these people"]
# Getting unique words
words = set()
for sentence in sentences:
	word = sentence.split()
	for w in word:
		words.add(w)
print (words)						 

words_list = list(words)

n = len(words_list)

co_occuranceMatrix = np.zeros((n,n))
for sentence in sentences:
	w = sentence.split()
	for i in range(0, len(w)):
		idx = words_list.index(w[i])
		if (i==0):
			co_occuranceMatrix[idx][words_list.index(w[i+1])] +=1
			continue
		if (i==len(w)-1):
			co_occuranceMatrix[idx][words_list.index(w[i-1])] +=1	
			continue

		co_occuranceMatrix[idx][words_list.index(w[i+1])] +=1
		co_occuranceMatrix[idx][words_list.index(w[i-1])] +=1	

print (co_occuranceMatrix, co_occuranceMatrix.shape)

# Singular Value Decomposition.
## Factorizes the matrix a into two unitary matrices U and Vh, and a 1-D array s of singular values (real, non-negative) such that a == U*S*Vh, 
## where S is a suitably shaped matrix of zeros with main diagonal s.
U, s, Vh = np.linalg.svd(co_occuranceMatrix, full_matrices=True)

# Plot and save 
for i in range(len(words_list)):
	fig = plt.gcf()
	fig.set_size_inches(18,10)
	plt.text(U[i, 0], U[i, 1],words_list[i])
	plt.ylim((-1, 1))
	plt.xlim((-1, 1))

plt.savefig('vis.jpg')