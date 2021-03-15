import time
import pickle

import numpy as np
import random
from collections import Counter

import spacy
import torch

from torchtext.data import Field
from torchtext.data import Bucketiterator
from torchtext.data import TabularDataset

from torchtext.datasets import snli

inputs = Field(lower=True, tokenize='spacy', batch_first=True)
answers = Field(sequential=False)

train_data, validation_data, test_data = snli.split(fields=(inputs, answers))
