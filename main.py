from __future__ import unicode_literals, print_function, division
from io import open
import string

from torch import optim
from preprocessing import prepare_data

#Preprocessing Data
input_language, output_language, pairs = prepare_data('eng', 'fra', True)