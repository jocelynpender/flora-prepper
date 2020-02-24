from data.make_helper_functions import *
import numpy as np
import pandas as pd


def read_fm_csv(fm_filepath):
    """Import Flora of Manitoba data for model training and perform processing tasks.
    :param:
        Filepath
    :return:
        Pandas data frame."""
    fm = pd.read_csv(fm_filepath, index_col=0,
                      dtype={"classification": "category", "text": np.object, "length": "int64"})
    assert fm is not None, 'FM file must contain data'
    return fm


def make_fm_data_frame(fm_filepath="data/raw/fm.csv",
                       frac_to_sample=0.1, balance_categories=False,
                       categories_to_keep=["key", "morphology", "taxon_identification", "distribution"]):
    fm = read_fm_csv(fm_filepath)
    trimmed_fm = trim_categories(fm, categories_to_keep)
    sampled_fm = sample_flora(trimmed_fm, frac_to_sample, balance_categories)
    return sampled_fm
