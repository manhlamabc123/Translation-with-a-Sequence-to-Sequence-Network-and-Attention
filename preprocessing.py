import unicodedata
import re
from language import Language

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