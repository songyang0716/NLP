import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchtext.legacy as legacy
from model import encoder1, encoder2, attention


