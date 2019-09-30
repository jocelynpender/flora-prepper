import numpy as np
import pandas as pd


def sample_flora(flora, frac_to_sample, balance_categories):
    """Trim down the dataset for development purposes with seed"""
    if balance_categories:  # https://stackoverflow.com/questions/45839316/pandas-balancing-data
        groups = flora.groupby('classification')
        flora = groups.apply(lambda x: x.sample(groups.size().min(), random_state=1))
        flora = flora.reset_index(drop=True)
        assert len(set(flora.groupby('classification').count().text)) == 1, 'All categories must have the same count'

    sampled_flora = flora.sample(frac=frac_to_sample, random_state=1)
    if frac_to_sample < 1:
        assert len(sampled_flora) < len(flora), 'Sampled dataset must be smaller than complete dataset'

    return sampled_flora