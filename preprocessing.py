import unicodedata
import re

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
    pass