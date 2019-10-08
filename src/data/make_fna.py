from src.data.make_helper_functions import *
import numpy as np
import pandas as pd


def read_fna_csv(fna_filepath):
    """Import Flora of North America (FNA) data for model training and perform processing tasks.
    Input:
        Filepath
    Output:
        Pandas data frame."""
    fna = pd.read_csv(fna_filepath, index_col=0,
                      dtype={"classification": "category", "text": np.object, "length": "int64"})
    assert fna is not None, 'FNA file must contain data'
    return fna


def make_fna_data_frame(fna_filepath="../data/external/fna_with_habitat.csv",
                        frac_to_sample=0.1, balance_categories=False,
                        categories_to_keep=["key", "morphology", "taxon_identification", "distribution", "habitat"],
                        rename_habitat=True):
    """Run all Flora of North America (FNA) processing tasks"""
    fna = read_fna_csv(fna_filepath)
    trimmed_fna = trim_categories(fna, categories_to_keep)
    if rename_habitat:
        trimmed_fna.loc[trimmed_fna.classification == "habitat", "classification"] = "distribution"
        trimmed_fna.classification.cat.remove_unused_categories(inplace=True)
    sampled_fna = sample_flora(trimmed_fna, frac_to_sample, balance_categories)
    return sampled_fna
