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


def process_text(text, tokenized_stop_words, to_lower=False, top_words=None):
    """This function is used to mimic nltk processing when visualizing or otherwise viewing data. This is not linked
    to the nltk vectorizer object"""
    processed_text = flora_tokenizer(text)  # Tokenize
    processed_text_no_stop_words = [word for word in processed_text if word.lower() not in tokenized_stop_words]

    if top_words:  # remove all but top words
        processed_text_no_stop_words = [word for word in processed_text_no_stop_words if word.lower() in top_words]
    if to_lower:  # lowercase if desired
        processed_text_no_stop_words = [word.lower() for word in processed_text_no_stop_words]

    return processed_text_no_stop_words


def process_text_tokenize_detokenize(flora_data_frame_text, tokenized_stop_words):
    """Pass it the text column of a data frame and it processes texts and throws it all back together."""
    processed_flora_data_series = flora_data_frame_text.apply(lambda x: process_text(x, tokenized_stop_words))
    processed_flora_data_series = processed_flora_data_series.apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    return processed_flora_data_series


def find_most_frequent_words(text_string, threshold=2000):
    """I'm finding that the upper limit of FreqDist.most_common() is 15,499"""
    assert type(text_string) == list, 'Input must be a list of words'
    text_string_lower = [word.lower() for word in text_string] # ensure all words are processed in lower case
    all_words = nltk.FreqDist(text_string_lower)
    word_lengths = [len(length) for length in list(all_words)]
    assert statistics.mean(word_lengths) > 1, 'Letters returned instead of words'
    most_freq_words = all_words.most_common()[:threshold]
    most_freq_words = [word[0] for word in most_freq_words] # Grab the first element of each tuple
   # assert len(most_freq_words) == threshold, 'Top words length does not match threshold'
    return most_freq_words


def filter_data_frame_top_words(flora_data_frame, top_words_text, tokenized_stop_words):
    """Transform the text column by removing words not in the set of top words passed to the function."""
    top_words_flora_text_series = flora_data_frame.text.apply(lambda x: process_text(x, tokenized_stop_words, to_lower=True, top_words=top_words_text))
    top_words_flora_data_frame = flora_data_frame.copy()
    top_words_flora_data_frame.text = top_words_flora_text_series.apply(lambda x: TreebankWordDetokenizer().detokenize(x))
    return top_words_flora_data_frame

