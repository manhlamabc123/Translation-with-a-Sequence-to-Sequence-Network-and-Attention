import unicodedata
import re
from language import Language
from constants import MAX_LENGTH

english_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

# Lowercase, trim, and remove non-letter characters
def unicode_to_ascii(string):
    return ''.join(
        character for character in unicodedata.normalize('NFD', string)
        if unicodedata.category(character) != 'Mn'
    )

def normalize_string(string):
    string = unicode_to_ascii(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    return string

def read_languages(language1, language2, reverse = False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (language1, language2), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(string) for string in line.split('\t')] for line in lines]

    # Reverse pairs, make Language instances
    if reverse:
        pairs = [list(reversed(pair)) for pair in pairs]
        input_language = Language(language1)
        output_language = Language(language2)
    else:
        input_language = Language(language1)
        output_language = Language(language2)

    return input_language, output_language, pairs

def filter_pair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH and pair[1].startswith(english_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(language1, language2, reverse=False):
    input_language, output_language, pairs = read_languages(language1, language2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_language.add_sentence(pair[0])
        output_language.add_sentence(pair[1])
    print("Counted words:")
    print(input_language.name, input_language.n_words)
    print(output_language.name, output_language.n_words)
    return input_language, output_language, pairs