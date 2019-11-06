from src.data.make_helper_functions import *
import pandas as pd
import numpy as np


def read_bc_csv(bc_filepath, first_col_index=False):
    """Import Illustrated Flora of BC CSV, scraped from E-Flora BC, for model training and perform processing tasks"""
    bc_index_col = 0 if first_col_index else False  # Version of the file has index integrity problems
    bc = pd.read_csv(bc_filepath, index_col=bc_index_col,
                     dtype={"species": np.object, "classification": "category", "content": np.object,
                            "morphology_content": np.object,
                            "morphology_content_misc": np.object})  # Collapse morphology columns into one
    assert bc is not None, 'E-Flora BC file must contain data'
    return bc


def merge_bc_columns(bc):
    """:param:
        Idiosyncractic E-Flora BC data frame with scraped data (i.e., three columns holding content:
    content, morphology_content, morphology_content_misc).

    :return:
        A data frame with collapsed text, with the three content columns removed, replaced by a column named text.
        Replace NA values with whitespace. (Could be improved).
        Due to string collapse, some elements may contain two whitespaces and nothing else."""
    bc[["content", "morphology_content", "morphology_content_misc", "species"]] \
        = bc[["content", "morphology_content", "morphology_content_misc", "species"]].fillna(value="")
    bc = bc[bc.classification.index.isna() == False]
    bc = bc[bc.classification.isna() == False]  # Drop columns with no classification
    assert bc.isna().any().sum() == 0, 'nan has not been removed'
    bc["text"] = bc["content"].map(str) + " " + bc["morphology_content"].map(str) + " " \
                 + bc["morphology_content_misc"].map(str)
    bc.drop(columns=["content", "morphology_content", "morphology_content_misc"], inplace=True)
    assert len(bc.columns) == 3, 'Reformatted data frame does not have three columns'
    return bc # BC data frame has artefacts of the scrape. TODO deal with this.

def make_bc_data_frame(bc_filepath="data/external/eflora-bc-full_no-id.csv", frac_to_sample=1,
                       balance_categories=True, categories_to_keep=["key", "morphology", "taxon_identification",
                                                                    "habitat", "discussion"]):
    """Perform requisite import tasks for E-Flora BC Data.
    :param:
        File path
    :return:
        Data frame with correctly named columns, trimmed and balanced dataset, and text length added to the dataframe
        for use in model building."""
    bc = read_bc_csv(bc_filepath)  # , first_col_index=True)
    bc = merge_bc_columns(bc)
    bc.classification.cat.rename_categories({'keys': 'key', 'name': 'taxon_identification'}, inplace=True)
    bc_with_length = add_length_to_data_frame(bc)
    bc_trimmed = trim_categories(bc_with_length, categories_to_keep)
    sampled_bc = sample_flora(bc_trimmed, frac_to_sample, balance_categories)
    return sampled_bc
