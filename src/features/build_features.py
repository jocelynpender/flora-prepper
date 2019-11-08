# also encode in the file the option to produce an tf/idf not just dtm
# tokenizer to remove unwanted elements from out data like symbols and numbers
#  https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
import logging
import sys
import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from build_stop_words import *
from process_text import *

#from features import build_stop_words, process_text


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


def build_train_test_split(text_counts, data_frame):
    """
    Build the training and test sets with some default arguments.
    :param text_counts: The csr_matrix DTM or TFIDF object
    :param data_frame: The entire flora_data_frame with a classification column
    :return: the training/testing split objects
    """
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data_frame['classification'], test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test

def main(path_to_flora_data_frame, path_to_dump):
    """This is my default feature build with no cleaning regime and no length features."""
    flora_data_frame = pd.read_csv(path_to_flora_data_frame, index_col=0)
    tokenized_stop_words = prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                                 "basionym", "source",
                                                                 "note", "notes", "morphology", "fna_id"])
    custom_vec, dtm_text_counts = build_dtm_text_counts(flora_tokenizer, tokenized_stop_words,
                                                        flora_data_frame)
    file_names = [path_to_dump + file_name for file_name in ["custom_vec", "dtm_text_counts"]]
    joblib.dump(custom_vec, file_names[0])
    joblib.dump(dtm_text_counts, file_names[1])

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    path_to_flora_data_frame = sys.argv[1]
    path_to_dump = sys.argv[2]
    main(path_to_flora_data_frame, path_to_dump)
