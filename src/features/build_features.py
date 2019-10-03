# also encode in the file the option to produce an tf/idf not just dtm
# tokenizer to remove unwanted elements from out data like symbols and numbers
#  https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize.treebank import TreebankWordDetokenizer
from src.data.misc import *


def flora_tokenizer(input_string):
    """Apply custom tokenization to fna strings.

    Notes
    -----

    Things to implement:
    remove stopwords
    remove 's
    remove numbers
    remove short words
    remove cells with two spaces (length = 2)
    """
    assert type(input_string) == str, 'Text column not string format'
    input_string = word_tokenize(input_string)
    assert type(input_string) == list, 'Tokens not returned in text column as list'
    return input_string


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


def build_dtm_text_counts(flora_tokenizer, tokenized_stop_words, data_frame):
    custom_vec = CountVectorizer(lowercase=True, tokenizer=flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(data_frame['text'])  # Build Document-Term-Matrix
    return text_counts


def build_tfidf_text_counts(flora_tokenizer, tokenized_stop_words, data_frame):
    custom_vec = TfidfVectorizer(lowercase=True, tokenizer=flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(data_frame['text'])  # Build TF-IDF Matrix
    return text_counts


def build_train_test_split(text_counts, data_frame):
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data_frame['classification'], test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test


def process_text(text, tokenized_stop_words):
    "This function is used to mimic nltk processing when visualizing or otherwise viewing data."
    processed_text = flora_tokenizer(text)  # Tokenize
    processed_text = [word for word in processed_text if word.lower() not in tokenized_stop_words]
    return processed_text


def process_text_tokenize_detokenize(flora_data_frame_text, tokenized_stop_words):
    """Pass it the text column of a data frame and it processes texts and throws it all back together."""
    processed_flora_data_series = flora_data_frame_text.apply(lambda x: process_text(x, tokenized_stop_words))
    processed_flora_data_series = processed_flora_data_series.apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    return processed_flora_data_series


def locate_empty_strings(flora_data_frame_text):
    """Takes a pandas series and return index to use for data frame drop operation."""
    assert type(flora_data_frame_text) == pd.core.series.Series, 'Input is not a pandas Series'
    flora_data_frame_text = flora_data_frame_text.map(lambda x: x.strip())  # convert all whitespace to nothing to
    # subsequently test and drop, https://stackoverflow.com/questions/2405292/how-to-check-if-text-is-empty-spaces
    # -tabs-newlines-in-python
    indx = flora_data_frame_text.map(is_blank) == False
    return indx


def process_length_in_place(flora_data_frame, tokenized_stop_words):
    before_process_length = flora_data_frame.text.apply(len)

    # Applying the same text processing used in the DTM/TFIDF models
    flora_data_frame.text = process_text_tokenize_detokenize(flora_data_frame.text, tokenized_stop_words)

    # Remove strings with no textual data
    flora_data_frame_no_empty = flora_data_frame[locate_empty_strings(flora_data_frame.text)]
    assert flora_data_frame_no_empty.shape[0] < flora_data_frame.shape[0], 'Rows with empty text strings not removed'
    after_process_length = flora_data_frame_no_empty.text.apply(len)
    assert sum(after_process_length) < sum(before_process_length), 'Text not processed'

    # Add new length data to data frame
    length_processed_flora_data_series = pd.concat([flora_data_frame_no_empty.text, after_process_length.rename('length')], axis=1)
    flora_data_frame_no_empty = flora_data_frame_no_empty.drop(columns='length')
    flora_data_frame_no_empty = flora_data_frame_no_empty.drop(columns='text')
    flora_data_frame_no_empty = pd.concat([flora_data_frame_no_empty, length_processed_flora_data_series], axis=1)
    return flora_data_frame_no_empty

# Restrict features to frequent terms

# freq_terms <- findFreqTerms(fna_dtm_train, 5)
# saveRDS(freq_terms, file = "models/budds_freq_terms.Rds")

# fna_dtm_freq_terms_train <- fna_dtm_train[, freq_terms]
# fna_dtm_freq_terms_test <- fna_dtm_test[, freq_terms]

# fna_train <- apply(fna_dtm_freq_terms_train, 2, yes.no)
# fna_test <- apply(fna_dtm_freq_terms_test, 2, yes.no)
