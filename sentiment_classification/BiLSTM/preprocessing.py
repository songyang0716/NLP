"""
Preprocess the movie reviews sentences
"""

import sys
import numpy as np
import random



# parse raw text into list (sentence in strings)
def parse_file2list(filename):
	contents = []
	with open(filename, 'r', encoding = "ISO-8859-1") as f:
		contents = [line.split("\n")[0] for line in f]
	print ("The file has lines: {}".format(len(contents)))
	return contents

# parse segmented corpus into list (sentence in list of words)
def parse_file2lol(filename):
	with open(filename, 'r', encoding = "ISO-8859-1") as f:
		contents = [string2list(line) for line in f]
	print ("The file has lines: {}".format(len(contents)))
	return contents


def string2list(sentence_in_string):
	"""convert strings with '\n' to list of words without '\n' """
	return sentence_in_string.strip().split()   # remove last \n

def shuffle(lol, seed=888):
	"""
	lol : list of list as input, in our example, lol is a list of sentences & labels 
		  lol[0] is sentences, lol[1] is labels
	seed : seed the shuffling
	
	shuffle inplace each list in the same order
	"""
	for l in lol:
		random.seed(seed)
		random.shuffle(l)



def processing():
	input_dir = "./data/"
	output_dir = input_dir
	# read sentences & labels
	data = [] 
	files = ["MR.task.train",
			 "MR.task.test"]
	for file in files:
		# sentences, labels
		sentences = parse_file2lol(input_dir + file + ".sentences")
		labels = parse_file2list(input_dir + file + ".labels")
		data.append([sentences, labels])
	
	assert len(data[0][0]) == len(data[0][1])
	assert len(data[1][0]) == len(data[1][1])
	
	# split the dataset: train, test
	shuffle(data[0], seed=888)
	train = data[0]
	test = data[1]
	# assume we don't have validation set, just use test
	valid = test

	assert len(train[0]) == len(train[1])
	assert len(valid[0]) == len(valid[1])
	assert len(test[0])  == len(test[1])

	raw_data = {"training": train,
				"validation": valid,
				"test": test}

	print(raw_data)

if __name__ == '__main__':
	processing()

