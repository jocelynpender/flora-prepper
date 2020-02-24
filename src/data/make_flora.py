from data.make_helper_functions import *
import numpy as np
import pandas as pd


def read_flora_csv(flora_filepath):
    """Import Flora of North America (FNA) data, Flora of Manitoba (FM) data, etc. for model training and perform
    processing tasks.
    :param:
        Filepath
    :return:
        Pandas data frame."""
    flora = pd.read_csv(flora_filepath, index_col=0,
                      dtype={"classification": "category", "text": np.object, "length": "int64"})
    assert flora is not None, 'flora file must contain data'
    return flora


def make_flora_data_frame(flora_filepath="data/raw/fna_with_habitat.csv",
                        frac_to_sample=0.1, balance_categories=False,
                        categories_to_keep=["key", "morphology", "taxon_identification", "distribution", "habitat"],
                        rename_habitat=True):
    """Run all Flora of North America (FNA) processing tasks"""
    flora = read_flora_csv(flora_filepath)
    trimmed_flora = trim_categories(flora, categories_to_keep)
    if rename_habitat:
        trimmed_flora.loc[trimmed_flora.classification == "distribution", "classification"] = "habitat"
        trimmed_flora.classification.cat.remove_unused_categories(inplace=True)
    sampled_flora = sample_flora(trimmed_flora, frac_to_sample, balance_categories)
    return sampled_flora
