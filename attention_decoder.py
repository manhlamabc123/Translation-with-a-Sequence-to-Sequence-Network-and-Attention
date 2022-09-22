import torch.nn as nn
from constants import MAX_LENGTH, device
import torch
import torch.nn.functional as F

class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_probability=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attention_weight = F.softmax(self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attention_applied = torch.bmm(attention_weight.unsqueeze(0), encoder_output.unsqueeze(0))

        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attention_weight

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)