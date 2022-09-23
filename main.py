from __future__ import unicode_literals, print_function, division
from evalute.evalute import evaluate_randomly

from preprocessing.preprocessing import prepare_data
from model.encoder import EncoderRNN
from constants import DROPOUT_PROBABILITY, HIDDEN_SIZE, N_ITERS, PRINT_EVERY, device
from model.attention_decoder import AttentionDecoderRNN
from train.train import train_iters

# Preprocessing Data
input_language, output_language, pairs = prepare_data('eng', 'fra', True)

# Training and Evaluating
encoder1 = EncoderRNN(input_language.n_words, HIDDEN_SIZE).to(device)
attention_decoder1 = AttentionDecoderRNN(HIDDEN_SIZE, output_language.n_words, dropout_probability=DROPOUT_PROBABILITY).to(device)

train_iters(input_language, output_language, pairs, encoder1, attention_decoder1, N_ITERS, print_every=PRINT_EVERY)

evaluate_randomly(input_language, output_language, pairs, encoder1, attention_decoder1)