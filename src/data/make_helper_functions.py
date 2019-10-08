import pandas as pd


def trim_categories(flora, categories_to_keep):
    """Only keep categories relevant to global model
    Input:
        A Pandas data frame of flora data, with the column of interest named "classification" and of type category.
        A set of categories to keep of type list.
    Output:
        A pandas data frame with rows belonging to unwanted categories dropped.
    """
    flora.classification.cat.set_categories(categories_to_keep, inplace=True)
    assert len(flora.classification.cat.categories) \
        == len(categories_to_keep), 'Number of new categories must match number of categories to keep'
    trimmed_fna = flora.dropna() # Remove NaN introduced by category filtering
    return trimmed_fna


def sample_flora(flora, frac_to_sample, balance_categories):
    """Trim down the dataset for development purposes with seed.

    Input:
        A Pandas data frame of flora data, with the column of interest named "classification".
    Output:
        A pandas data frame, shuffled, sampled & balanced."""
    if balance_categories:  # https://stackoverflow.com/questions/45839316/pandas-balancing-data
        groups = flora.groupby('classification')
        flora = groups.apply(lambda x: x.sample(groups.size().min(), random_state=1))
        flora = flora.reset_index(drop=True)
        assert len(set(flora.groupby('classification').count().text)) == 1, 'All categories must have the same count'

    sampled_flora = flora.sample(frac=frac_to_sample, random_state=1) # This shuffles the dataset
    if frac_to_sample < 1:
        assert len(sampled_flora) < len(flora), 'Sampled dataset must be smaller than complete dataset'

    return sampled_flora


def add_length_to_data_frame(data_frame):
    """Returns a data frame with length of the text column calculated.
    Input:
        A Pandas data frame of flora data, with the text column named "text".
    Returns:
        The data frame with a new column: 'length'."""
    length = data_frame['text'].apply(len)
    data_frame_with_length = pd.concat([data_frame, length.rename('length')], axis=1)
    return data_frame_with_length
