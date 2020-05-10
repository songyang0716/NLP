################################################################################################################
#### Reference: https://github.com/blackredscarf/pytorch-SkipGram/blob/master/word2vec.py
################################################################################################################
class Word2Vec:
	def __init__(self, path, vocab_size, embedding_size=100, learning_rate=1.0):
		self.corpus = read_own_data(path)
		self.data, self.word_count, self.word2index, self.index2word = build_dataset(self.corpus, vocab_size)
		self.model = SkipGramNeg(vocabulary_size, embedding_size).cuda()
		self.model_optim = SGD(self.model.parameters(), lr=learning_rate)

	def train(self)