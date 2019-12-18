import pandas as pd
import numpy as np

budds_results = pd.read_csv("flora_commons_workflow/budds_results_to_examine_rekey.csv", index_col=0)
budds_results.reclassification[0] = budds_results.classification[0]  # Fix first item


# TODO: find schema document from Github

def merge_classification_blocks(classifier_results, classification_results_column):
    """
    With a dataframe of classification results, merge text blocks together and return a dataframe with merged strings
    and classification types.
    :param:
        Classifier results
    :return:
    """
    # https://stackoverflow.com/questions/14358567/finding-consecutive-segments-in-a-pandas-data-frame
    assert type(classifier_results) == pd.core.frame.DataFrame, "Input data must be a DataFrame"
    assert type(classifier_results['text'][0]) == str, "Column named text must contain strings of classified text"

    classifier_results['block'] = (
                classifier_results.reclassification.shift(1) != classifier_results.reclassification).astype(
        int).cumsum()
    group_blocks = classifier_results.groupby('block')
    runs = group_blocks['text'].apply(np.array)  # slow
    strings = runs.apply(lambda x: '\n'.join(x)).reset_index(drop=True)

    runs_classification = group_blocks.first().iloc[:, classification_results_column].reset_index(drop=True)
    strings_classification = pd.concat([strings, runs_classification], axis=1)
    assert len(strings_classification) == len(strings), "Mismatch in length of merged strings & classifications"
    assert len(strings_classification) == len(
        runs_classification), "Mismatch in length of merged strings & classifications"

    return strings_classification


open_tags = {'taxon_identification': '<taxon_identification status="ACCEPTED">',
             'morphology': '<description type="morphology">',
             'habitat': '<description type="habitat">',
             'key': '<key>'
             }
close_tags = {'taxon_identification': '</taxon_identification>',
              'morphology': '</description>',
              'habitat': '</description>',
              'key': '</key>'
              }


def add_tags(row, open_schema_dict, close_schema_dict):
    open = open_schema_dict[row[1]]
    close = close_schema_dict[row[1]]
    tag_text = row.text.join([open, close])
    return tag_text


def add_schema(strings_classification, open_schema_dict, close_schema_dict):
    """
    Take classified and merged strings from classification blocks and return a document with tags surrounding the strings.
    """
    document = strings_classification.apply(lambda x: add_tags(x, open_schema_dict, close_schema_dict), axis=1)
    document = '\n'.join(document)
    return document


budds_strings_classification = merge_classification_blocks(budds_results, 2)
budds_document = add_schema(budds_strings_classification, open_tags, close_tags)
text_file = open("budds_document.txt", "wt")
n = text_file.write(budds_document)
text_file.close()