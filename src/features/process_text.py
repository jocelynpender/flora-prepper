import nltk
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import statistics


def flora_tokenizer(input_string):
    """Apply custom tokenization to fna strings.

    TODO:
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


def process_text(text, tokenized_stop_words, top_words=None):
    """This function is used to mimic nltk processing when visualizing or otherwise viewing data. This is not linked
    to the nltk vectorizer object"""
    processed_text = flora_tokenizer(text)  # Tokenize
    processed_text_no_stop_words = [word for word in processed_text if word.lower() not in tokenized_stop_words]
    num_words_no_stop_words = len(processed_text_no_stop_words)
    assert num_words_no_stop_words < len(processed_text), 'Stop words not removed'
    if top_words:
        processed_text_no_stop_words = [word for word in processed_text_no_stop_words if word.lower() in top_words]
        assert len(processed_text_no_stop_words) < num_words_no_stop_words, 'Non-top words not removed'
    return processed_text_no_stop_words


def process_text_tokenize_detokenize(flora_data_frame_text, tokenized_stop_words):
    """Pass it the text column of a data frame and it processes texts and throws it all back together."""
    processed_flora_data_series = flora_data_frame_text.apply(lambda x: process_text(x, tokenized_stop_words))
    processed_flora_data_series = processed_flora_data_series.apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    return processed_flora_data_series
