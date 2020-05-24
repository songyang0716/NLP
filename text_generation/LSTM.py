import sys
import torch
import torch.optim as optim
import random
import numpy as np
import spacy
from collections import Counter
from nltk.tokenize import word_tokenize



# Parameters
train_file='yelp_review_10000.txt'
seq_size=32
batch_size=50
embedding_size=64
lstm_size=64
gradients_norm=5
predict_top_k=5




def read_file(train_file, batch_size, seq_size):
	with open(train_file, "r", encoding='utf-8') as f:
		reviews = f.read()
	words = reviews.split()
	words_freq = Counter(words)
	words_freq = sorted(words_freq.items(), key=lambda x: x[1], reverse=True)

	vocab = set(words)
	word_size = len(words)
	vocab_size = len(vocab)
	word_to_idx = {v:i for i, v in enumerate(vocab)}
	idx_to_word = {i:v for i, v in enumerate(vocab)}

	print('Vocabulary size', vocab_size)

	return
	# print(len(reviews))
	# return 


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# nlp = spacy.load("en_core_web_sm")

	read_file(train_file, batch_size, seq_size)

if __name__ == '__main__':
    main()

