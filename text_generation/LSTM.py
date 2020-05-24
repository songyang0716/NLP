# Run the whole thing using google Colab #

import sys
import torch
import torch.optim as optim
import random
import numpy as np
import spacy
from collections import Counter
from nltk.tokenize import word_tokenize
from model import LSTM 



# Parameters
train_file = 'yelp_review_10000.txt'
seq_size = 32
batch_size = 16
emb_size = 100
hidden_size = 64
gradients_norm = 5
predict_top_k = 5
output_sentence_length = 50
l_rate = 0.01
epoch = 1



def read_file(train_file, batch_size, seq_size):
	with open(train_file, "r", encoding='utf-8') as f:
		reviews = f.read()
	words = reviews.split()
	words = [word.lower() for word in words]
	words_freq = Counter(words)
	words_freq = sorted(words_freq.items(), key=lambda x: x[1], reverse=True)

	vocab = set(words)
	word_size = len(words)
	vocab_size = len(vocab)
	word_to_idx = {v:i for i, v in enumerate(vocab)}
	idx_to_word = {i:v for i, v in enumerate(vocab)}

	word_index = [word_to_idx[w] for w in words]
	num_batches = int(len(word_index) / (seq_size * batch_size))

	in_text = word_index[:num_batches * batch_size * seq_size]
	out_text = np.zeros_like(in_text)

	out_text[:-1] = in_text[1:]
	out_text[-1] = in_text[0]

	in_text = np.reshape(in_text, (num_batches,-1))
	out_text = np.reshape(out_text, (num_batches,-1))

	return idx_to_word, word_to_idx, vocab_size, in_text, out_text


def generate_batch(in_text, out_text, batch_size, seq_size):
	num_batches = in_text.shape[0]
	for i in range(num_batches):
		in_text_cur = np.reshape(in_text[i], (batch_size, seq_size))
		out_text_cur =  np.reshape(out_text[i], (batch_size, seq_size))
		yield in_text_cur, out_text_cur


def predict(device, model, vocab_size, word_to_idx, idx_to_word, top_k=5):
	model.eval()
	words = ['i', 'like']
	h0, c0 = model.initial_state(1)
	h0 = h0.to(device)
	c0 = c0.to(device)
	for w in words:
		ix = torch.tensor([[word_to_idx[w]]]).to(device)
		output, (h0, c0) = model(ix, (h0, c0))
	output = output.squeeze()
	_, top_ix = torch.topk(output, k=top_k)

	choices = top_ix.detach().cpu().numpy()
	choice = np.random.choice(choices,1)[0]

	words.append(idx_to_word[choice])

	for _ in range(output_sentence_length):
		ix = torch.tensor([[choice]]).to(device)
		output, (h0, c0) = model(ix, (h0, c0))
		output = output.squeeze()
		_, top_ix = torch.topk(output, k=top_k)
		choices = top_ix.detach().cpu().numpy()
		choice = np.random.choice(choices,1)[0]
		words.append(idx_to_word[choice])

	print(' '.join(words).encode('utf-8'))

	return ""


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	idx_to_word, word_to_idx, vocab_size, in_text, out_text = read_file(train_file, batch_size, seq_size)

	lstm_model = LSTM(vocab_size, seq_size, emb_size, hidden_size)
	lstm_model = lstm_model.to(device)

	lstm_optim = optim.Adam(lstm_model.parameters(), lr=l_rate)
	loss_function = torch.nn.CrossEntropyLoss()


	for i in range(epoch):
		batches = generate_batch(in_text, out_text, batch_size, seq_size)
		h0, c0 = lstm_model.initial_state(batch_size)
		h0 = h0.to(device)
		c0 = c0.to(device)
		total_loss, iterations = 0, 0

		for x, y in batches:
			# shape of x is (batch_size, seq_size)
			x = torch.tensor(x).to(device)
			y = torch.tensor(y).to(device)

			lstm_optim.zero_grad()

			logits, (h0, c0) = lstm_model(x, (h0, c0))

			_,_,n_cat = logits.shape

			loss = loss_function(logits.view(-1, n_cat), y.view(-1))
			
			print(loss.item())
			total_loss += loss.item()
			iterations += 1
			loss.backward()

       		# Starting each batch, we detach the hidden state from how it was previously produced.
			# If we didn't, the model would try backpropagating all the way to start of the dataset.
			h0 = h0.detach()
			c0 = c0.detach()

			_ = torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), gradients_norm)
			lstm_optim.step()
			# break

		avg_loss = total_loss / iterations

		print('Epoch: {}'.format(i),
				  'Loss: {}'.format(avg_loss))
		if i % 10 == 0:
			torch.save(lstm_model.state_dict(),'checkpoint_pt/model-{}.pth'.format(i))		
		
	_ = predict(device, lstm_model, vocab_size, word_to_idx, idx_to_word, top_k=predict_top_k)


if __name__ == '__main__':
	main()

