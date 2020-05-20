import sys
import torch
import torch.optim as optim
import random
import numpy as np
import spacy
from nltk.tokenize import word_tokenize
from model import GloVe


# Parameters
l_rate = 0.001
num_epochs = 100
batch_size = 20
alpha = 0.75
context_size = 5
window_size = 10
embed_size = 2
xmax = 100
n_reviews = 5000



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


comat = torch.from_numpy(comat)
print(comat.shape)

# non-zero occurence index
# coocs = np.transpose(np.nonzero(comat))
model = GloVe(comat, embed_size)
model_optim = optim.Adam(model.parameters(), lr=l_rate)





def get_batch(batch_idx):
	# word_to_idx[wordtokens[i]]
	c_word_idx = []
	context_word_idx = []
	for i in range(batch_idx*batch_size, (batch_idx+1)*batch_size, 1):
		for j in range(1, window_size+1):
			ind = word_to_idx[wordtokens[i]]
		if i - j > 0:
			c_word_idx.append(ind)
			context_word_idx.append(word_to_idx[wordtokens[i-j]])
		if i + j < word_size:
			c_word_idx.append(ind)
			context_word_idx.append(word_to_idx[wordtokens[i+j]])


	return c_word_idx, context_word_idx

	# print(c_word_index)
	# return "",""

# train the model
for epoch in range(num_epochs):
	n_batches = int(word_size/batch_size)
	avg_loss = 0
	for j in range(n_batches):
		input_index, output_index = get_batch(j)
		input_index = torch.FloatTensor(input_index).to(torch.int64)
		output_index = torch.FloatTensor(output_index).to(torch.int64)
		model_optim.zero_grad()
		loss = model(input_index, output_index)
		loss.backward()
		model_optim.step()
		avg_loss += loss

	avg_loss = avg_loss/n_batches
	print(avg_loss)



# def gen_batch(coocs):
# 	# extract f
# 	sample = np.random.choice(np.arange(len(coocs)), size=batch_size, replace=False)
# 	l_vecs, r_vecs, covals, l_v_bias, r_v_bias = [],[],[],[]
# 	for chosen in sample:
# 		# a pair of index
# 		ind = tuple(coocs[chosen])
# 		l_vecs.append(l_embed[ind[0]])
# 		r_vecs.append(r_embed[ind[1]])
# 		covals.append(coocs[ind])
# 		l_v_bias.append(l_v_bias[ind[0]])
# 		r_v_bias.append(r_v_bias[ind[1]])
# 	return l_vecs, r_vecs, covals, l_v_bias, r_v_bias


# print(len(coocs))
# gen_batch()

# Helper functions
def show_similar_words(words, comat, top_n=3):
	for w in words:
		print(w)
		sww = comat[word_to_idx[w],:]
		sw = sww.argsort()[-top_n:][::-1]
		print([idx_to_word[i] for i in sw])





