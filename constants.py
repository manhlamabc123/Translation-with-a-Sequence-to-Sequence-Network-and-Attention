import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10

SOS_token = 0
EOS_token = 1