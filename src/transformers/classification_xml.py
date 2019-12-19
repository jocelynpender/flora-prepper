import pandas as pd
import numpy as np


def add_block_column(classifier_results, classification_results_column):
    """
    https://stackoverflow.com/questions/14358567/finding-consecutive-segments-in-a-pandas-data-frame

    """
    assert type(classifier_results) == pd.core.frame.DataFrame, "Input data must be a DataFrame"
    assert type(classifier_results['text'][0]) == str, "Column named text must contain strings of classified text"

    classifier_results['block'] = (
            classifier_results.iloc[:, classification_results_column].shift(1) != classifier_results.iloc[:,
                                                                                  classification_results_column]).astype(
        int).cumsum()  # Find runs of a particular classification
    return classifier_results


def build_text_runs(classifier_results):
    """

    """
    group_blocks = classifier_results.groupby('block')
    runs = group_blocks['text'].apply(np.array)  # slow

    return runs, group_blocks


def merge_classification_blocks(classifier_results, classification_results_column):
    """
    With a dataframe of classification results, merge text blocks together and return a dataframe with merged strings
    and classification types.
    :param:
        classifier_results: Classifier results
    :return:
        strings_classification: A dataframe with merged strings and classification types.
    """

    classifier_results = add_block_column(classifier_results, classification_results_column)
    runs, group_blocks = build_text_runs(classifier_results)
    strings = runs.apply(lambda x: '\n'.join(x)).reset_index(drop=True)
    runs_classification = group_blocks.first().iloc[:, classification_results_column].reset_index(drop=True)
    strings_classification = pd.concat([strings, runs_classification], axis=1)

    assert len(strings_classification) == len(strings), "Mismatch in length of merged strings & classifications"
    assert len(strings_classification) == len(
        runs_classification), "Mismatch in length of merged strings & classifications"

    return strings_classification


def add_tags(row, open_schema_dict, close_schema_dict):
    """For a given row in a dataframe, wrap it in its corresponding schema opening and closing tags
    :param:
    :return:
    """
    open = open_schema_dict[row[1]]
    close = close_schema_dict[row[1]]
    tag_text = row.text.join([open, close])
    return tag_text


def add_schema(strings_classification, open_schema_dict, close_schema_dict):
    """
    Take classified and merged strings from classification blocks and return a document with tags surrounding the
    strings for all data.
    """
    single_document = strings_classification.apply(lambda x: add_tags(x, open_schema_dict, close_schema_dict), axis=1)
    single_document = '\n'.join(single_document)
    return single_document


def write_file(document, file_name):
    """
    Simple helper function to write a text file
    """
    text_file = open(file_name, "wt")
    n = text_file.write(document)
    text_file.close()


def write_documents(strings_classification, open_tags, close_tags, prematter):
    whole_document = add_schema(strings_classification, open_tags, close_tags)

    # every time you encounter the prematter, split and write a file!
    sep_documents = whole_document.split(sep=prematter)
    file_names = [str(x) + ".xml" for x in range(1, len(sep_documents) + 1)]
    write_docs = [write_file(x, file_names[ind]) for ind, x in enumerate(sep_documents)]

    return write_docs
