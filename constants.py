import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10

SOS_token = 0
EOS_token = 1

TEACHER_FORCING_RATIO = 0.5

LEARNING_RATE = 0.01

HIDDEN_SIZE = 256

PRINT_EVERY = 5000

N_ITERS = 1

DROPOUT_PROBABILITY = 0.1