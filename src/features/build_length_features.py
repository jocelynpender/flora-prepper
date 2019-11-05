import pandas as pd
from scipy import sparse
from src.data.misc import is_blank
from src.features.process_text import process_text_tokenize_detokenize
import numpy as np


def locate_empty_strings(flora_data_frame_text):
    """Takes a pandas series and return index to use for data frame drop operation."""
    assert type(flora_data_frame_text) == pd.core.series.Series, 'Input is not a pandas Series'
    flora_data_frame_text = flora_data_frame_text.map(lambda x: x.strip())  # convert all whitespace to nothing to
    # subsequently test and drop, https://stackoverflow.com/questions/2405292/how-to-check-if-text-is-empty-spaces
    # -tabs-newlines-in-python
    indx = flora_data_frame_text.map(is_blank) == False
    return indx


def process_length_in_place(flora_data_frame, tokenized_stop_words):
    """Process text using the same text processing procedure as was used in the DTM/TFIDF models, and recreate the
    length column with the cleaned text strings. This results in a more accurate length metric.

    Returns:
    flora_data_frame with revised text length column and rows with only blanks or empty text
    strings removed."""
    before_process_length = flora_data_frame.text.apply(len)

    # Applying the same text processing used in the DTM/TFIDF models
    flora_data_frame.text = process_text_tokenize_detokenize(flora_data_frame.text, tokenized_stop_words)

    # Remove strings with no textual data
    flora_data_frame_no_empty = flora_data_frame[locate_empty_strings(flora_data_frame.text)]
    assert flora_data_frame_no_empty.shape[0] < flora_data_frame.shape[0], 'Rows with empty text strings not removed'
    after_process_length = flora_data_frame_no_empty.text.apply(len)
    assert sum(after_process_length) < sum(before_process_length), 'Text not processed'

    # Add new length data to data frame
    length_processed_flora_data_series = pd.concat(
        [flora_data_frame_no_empty.text, after_process_length.rename('length')], axis=1)
    flora_data_frame_no_empty = flora_data_frame_no_empty.drop(columns='length')
    flora_data_frame_no_empty = flora_data_frame_no_empty.drop(columns='text')
    flora_data_frame_no_empty = pd.concat([flora_data_frame_no_empty, length_processed_flora_data_series], axis=1)
    return flora_data_frame_no_empty


def prepare_length_features(text_counts, custom_vec, length_processed_flora_data_frame):
    """Instead of a sparse matrix of text counts, let's build a sparse matrix including text counts and length to
    train the model. """
    vocab = custom_vec.get_feature_names()  # https://stackoverflow.com/questions/39121104/how-to-add-another-feature
    # -length-of-text-to-current-bag-of-words-classificati

    length_model_data_frame = pd.DataFrame(text_counts.toarray(), columns=vocab)
    length_model_data_frame = pd.concat(
        [length_model_data_frame, length_processed_flora_data_frame['length'].reset_index(drop=True)], axis=1)

    length_model_data_frame_values = length_model_data_frame.values.astype(np.float64)
    length_model_sparse = sparse.csr_matrix(length_model_data_frame_values)

    assert length_model_sparse.shape > text_counts.shape, 'Length model should have one more column of data than BOW ' \
                                                          'model '
    return length_model_sparse
