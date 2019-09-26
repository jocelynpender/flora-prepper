from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def tokenize(fna):
    assert type(fna.text[0]) == str, 'Text column not string format'
    fna.text = fna.text.map(lambda x: word_tokenize(x))
    assert type(fna.text[0]) == list, 'Tokens not returned in text column as list'
    return fna


def build_nltk_stop_words(stop_words):
    """Takes a set of stop words and add nltk's english words to it"""
    import nltk
    nltk.download('stopwords')
    nltk_stop_words = set(stopwords.words('english'))
    assert len(nltk_stop_words) > 0, 'NLTK words not loaded'
    stop_words += nltk_stop_words
    return stop_words


def remove_stop_words(tokenized_fna, stop_words):
    """Returns a cleaned fna data matrix with no stop words"""
    count_words_pre_process = tokenized_fna.text.map(len).sum()  # Counter to test function is working properly
    remove_words = lambda x: [word for word in x if not word in stop_words]  # If word is not in stopwords, append to
    # list
    tokenized_fna.text = tokenized_fna.text.map(remove_words)
    count_words_post_process = tokenized_fna.text.map(len).sum()  # Counter to test function is working properly
    assert count_words_post_process < count_words_pre_process, 'No stopwords removed'
    return tokenized_fna


# Do feature cleaning tasks
def clean_fna(fna, fna_stop_words, do_remove_stop_words=True, include_nltk_stop_words=False):
    """Returns a cleaned fna data frame for use in feature generation. Option to remove nltk stop words"""
    tokenized_fna = tokenize(fna)
    if include_nltk_stop_words:
        fna_stop_words = build_nltk_stop_words(fna_stop_words)
    if do_remove_stop_words:
        tokenized_fna = remove_stop_words(tokenized_fna, fna_stop_words)
    return tokenized_fna

# options:
# stem, lemmatization,
# to lowercase (can be done in build_features), remove whitespace = not necessary
# check encoding??
# remove punctuation??
# remove brackets?
# keep numbers... for the keys!
# replace abbreviations e.g. fam with family?

# do we care about corpora in terms of a data structure in Python nltk???
