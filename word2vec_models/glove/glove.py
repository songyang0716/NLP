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
alpha = 3/4
context_size = 3
embed_size = 100
xmax = 2
n_reviews = 50



# Read file
nlp = spacy.load("en_core_web_sm")
f = open('../yelp_review_10000.txt', "r", encoding='utf-8')
reviews = f.readlines()
wordtokens = []
for review in reviews[:n_reviews]:
	#and not word in nlp.Defaults.stop_words
	tokens = [word.lower() for word in word_tokenize(review) if word.isalnum()]
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
comat = np.zeros((vocab_size, vocab_size))
for i in range(word_size):
	for j in range(1, context_size+1):
		ind = word_to_idx[wordtokens[i]]
		if i - j > 0:
			# left window - context_size words
			lind = word_to_idx[wordtokens[i-j]] 
			# provide some weight to the nearby words, the further, the smaller
			comat[ind, lind] += 1/j
		if i + j < word_size:
			# right window - context_size words
			rind = word_to_idx[wordtokens[i+j]]
			comat[ind, rind] += 1/j
print(comat.shape)


# non-zero occurence index
coocs = np.transpose(np.nonzero(comat))






# some helper functions
def show_similar_words(words, comat, top_n=3):
	for w in words:
		print(w)
		sww = comat[word_to_idx[w],:]
		sw = sww.argsort()[-top_n:][::-1]
		print([idx_to_word[i] for i in sw])

def wf(x):
	if x < xmax:
		return (x/xmax)**alpha
	else:
		return 1




