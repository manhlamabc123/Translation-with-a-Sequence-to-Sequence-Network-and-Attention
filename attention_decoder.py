import torch.nn as nn
from constants import MAX_LENGTH

class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_lengh=MAX_LENGTH):
        pass