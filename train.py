from language import Language
from constants import EOS_token, device

def indexes_from_sentence(language, sentence):
    return [language.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(language, sentence):
    indexes = indexes_from_sentence(language, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(pair, input_language, output_language):
    input_tensor = tensor_from_sentence(input_language, pair[0])
    target_tensor = tensor_from_sentence(output_language, pair[1])
    return (input_tensor, target_tensor)