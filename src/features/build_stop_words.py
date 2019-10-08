from nltk import word_tokenize
from nltk.corpus import stopwords


def build_nltk_stop_words(stop_words):
    """Takes a set of stop words and add nltk's english words to it"""
    import nltk
    nltk.download('stopwords')
    nltk_stop_words = set(stopwords.words('english'))
    assert len(nltk_stop_words) > 0, 'NLTK words not loaded'
    stop_words += nltk_stop_words
    return stop_words


def prepare_stop_words(custom_stop_words, include_nltk_stop_words=True):
    """Tokenize the stop words"""
    assert type(custom_stop_words) == list, 'Stop words not provided as a list'
    if include_nltk_stop_words:
        custom_stop_words = build_nltk_stop_words(custom_stop_words)
    tokenized_stop_words = []
    for word in custom_stop_words:
        tokenized_stop_words.extend(word_tokenize(word))
    tokenized_stop_words = frozenset(tokenized_stop_words)
    return tokenized_stop_words