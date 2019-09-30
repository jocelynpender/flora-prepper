from src.data.make_helper_functions import *
import numpy as np
import pandas as pd


def read_fna_csv(fna_filepath):
    """Import Flora of North America (FNA) data for model training and perform processing tasks"""
    fna = pd.read_csv(fna_filepath, index_col=0,
                      dtype={"classification": "category", "text": np.object, "length": "int64"})
    assert fna is not None, 'FNA file must contain data'
    return fna


def trim_categories(fna, categories_to_keep):
    """Only keep categories relevant to global model"""
    fna.classification.cat.set_categories(categories_to_keep, inplace=True)
    assert len(fna.classification.cat.categories) \
        == len(categories_to_keep), 'Number of new categories must match number of categories to keep'

    # Remove NaN introduced by category filtering
    trimmed_fna = fna.dropna()
    return trimmed_fna


def make_fna_data_frame(fna_filepath="data/external/fna_keys.csv",
                        frac_to_sample=0.1, balance_categories=False,
                        categories_to_keep=["key", "morphology", "taxon_identification"]):
    """Run all Flora of North America (FNA) processing tasks"""
    fna = read_fna_csv(fna_filepath)
    trimmed_fna = trim_categories(fna, categories_to_keep)
    sampled_fna = sample_flora(trimmed_fna, frac_to_sample, balance_categories)
    return sampled_fna

#fna = make_fna_data_frame(fna_filepath="../../data/external/fna_keys.csv")

