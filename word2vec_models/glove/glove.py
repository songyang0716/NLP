import sys
import torch
import torch.optim as optim
import random
import numpy as np
import spacy
from nltk.tokenize import word_tokenize



# Parameters
l_rate = 0.001
num_epochs = 10
batch_size = 20
alpha = 0.8
context_size = 3
embed_size = 100
xmax = 2
n_reviews = 100



# Read file
nlp = spacy.load("en_core_web_sm")
f = open('../yelp_review_10000.txt', "r", encoding='utf-8')
reviews = f.readlines()
wordtokens = []
for review in reviews[:n_reviews]:
	tokens = [word.lower() for word in word_tokenize(review)]
	#encode('utf-8')
	wordtokens.extend(tokens)

vocab = set(wordtokens)
word_size = len(wordtokens)
vocab_size = len(vocab)
word_to_idx = {v:i for i, v in enumerate(vocab)}
idx_to_word = {i:v for i, v in enumerate(vocab)}

# print(vocab)
# print(word_size)
# print(vocab_size)
# print(word_to_idx)


# cooccurence matrix
cmat = np.zeros((vocab_size, vocab_size))
for i in range(word_size):
	for j in range(1, context_size+1):
		ind = word_to_idx[wordtokens[i]]
		if i - j > 0:
			# left window - context_size words
			lind = word_to_idx[wordtokens[i-j]] 
			# provide some weight to the nearby words, the further, the smaller
			cmat[ind, lind] += 1/j
		if i + j < word_size:
			# right window - context_size words
			rind = word_to_idx[wordtokens[i+j]]
			cmat[ind, rind] += 1/j
print(cmat.shape)


# some words 613 menu   465 steak   345 manager
example_words = ['menu','steak','manager']
for w in example_words:
	print(w)
	sww = cmat[word_to_idx[w],:]
	# top 3 most similar words
	sw = sww.argsort()[-3:][::-1]
	print([idx_to_word[i] for i in sw])


