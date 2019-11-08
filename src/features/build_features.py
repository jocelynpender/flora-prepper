# also encode in the file the option to produce an tf/idf not just dtm
# tokenizer to remove unwanted elements from out data like symbols and numbers
#  https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_dtm_text_counts(flora_tokenizer, tokenized_stop_words, data_frame):
    """This function initializes an nltk vectorizer object and fits it to the text data to return a dtm for model
    training. The custom_vec flora_tokenizer allows us to swap in and out custom cleaning regimes.
    :param flora_tokenizer: the custom cleaning tokenizer
    :param tokenized_stop_words: the stop words cleaned with the same cleaning regime.
    :param data_frame: data to fit
    :return:
    The vectorizer which performs the DTM fit and the DTM or the sparse representation of the DTM counts using
    scipy.sparse.csr_matrix.
"""
    custom_vec = CountVectorizer(lowercase=True, tokenizer=flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(data_frame['text'])  # Build Document-Term-Matrix
    return custom_vec, text_counts


def build_tfidf_text_counts(flora_tokenizer, tokenized_stop_words, data_frame):
    """
    This function initializes an nltk tfidf vectorizer object and fits it to the text data to return a dtm for model
    training. The custom_vec flora_tokenizer allows us to swap in and out custom cleaning regimes.
    :param flora_tokenizer: the custom cleaning tokenizer
    :param tokenized_stop_words: the stop words cleaned with the same cleaning regime.
    :param data_frame: data to fit
    :return:
    The vectorizer which performs the DTM fit and the DTM or the sparse representation of the DTM counts using
    scipy.sparse.csr_matrix
    """
    custom_vec = TfidfVectorizer(lowercase=True, tokenizer=flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(data_frame['text'])  # Build TF-IDF Matrix
    return custom_vec, text_counts

