from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def fna_tokenize(fna_string):
    """Apply custom tokenization to fna strings"""
    assert type(fna_string) == str, 'Text column not string format'
    fna_string = word_tokenize(fna_string)
    assert type(fna_string) == list, 'Tokens not returned in text column as list'
    return fna_string


def build_nltk_stop_words(stop_words):
    """Takes a set of stop words and add nltk's english words to it"""
    import nltk
    nltk.download('stopwords')
    nltk_stop_words = set(stopwords.words('english'))
    assert len(nltk_stop_words) > 0, 'NLTK words not loaded'
    stop_words += nltk_stop_words
    return stop_words

def prepare_stop_words(fna_stop_words, include_nltk_stop_words = True):
    """Tokenize the stop words"""
    assert type(fna_stop_words) == list, 'Stop words not provided as a list'
    if include_nltk_stop_words:
        fna_stop_words = build_nltk_stop_words(fna_stop_words)
    tokenized_stop_words = []
    for word in fna_stop_words:
        tokenized_stop_words.extend(word_tokenize(word))
    tokenized_stop_words = frozenset(tokenized_stop_words)
    return tokenized_stop_words

# options:
# stem, lemmatization,
# to lowercase (can be done in build_features), remove whitespace = not necessary
# check encoding??
# remove punctuation??
# remove brackets?
# keep numbers... for the keys!
# replace abbreviations e.g. fam with family?

# do we care about corpora in terms of a data structure in Python nltk???
