import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field
from torchtext.data import BucketIterator

from torchtext.datasets import SNLI
