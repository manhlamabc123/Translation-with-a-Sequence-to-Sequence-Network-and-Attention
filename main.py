from __future__ import unicode_literals, print_function, division
from evalute import evaluate_randomly

from preprocessing import prepare_data
from encoder import EncoderRNN
from constants import HIDDEN_SIZE, device
from attention_decoder import AttentionDecoderRNN
from train import train_iters

# Preprocessing Data
input_language, output_language, pairs = prepare_data('eng', 'fra', True)

# Training and Evaluating
encoder1 = EncoderRNN(input_language.n_words, HIDDEN_SIZE).to(device)
attention_decoder1 = AttentionDecoderRNN(HIDDEN_SIZE, output_language.n_words, dropout_probability=0.1).to(device)

train_iters(encoder1, attention_decoder1, 75000, print_every=5000)

evaluate_randomly(encoder1, attention_decoder1)