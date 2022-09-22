from __future__ import unicode_literals, print_function, division
from io import open
import string
import random

from torch import optim
from preprocessing import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Preprocessing Data
input_language, output_language, pairs = prepare_data('eng', 'fra', True)