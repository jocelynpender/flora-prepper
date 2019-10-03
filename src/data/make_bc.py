from src.data.make_helper_functions import *
import numpy as np

def read_bc_csv(bc_filepath):
    """Import Flora of North America (FNA) data for model training and perform processing tasks"""
    bc = pd.read_csv(bc_filepath, index_col=0,
                     dtype={"species": np.object, "classification": "category", "content": np.object,
                            "morphology_content": np.object, "morphology_content_misc": np.object})
    assert bc is not None, 'E-Flora BC file must contain data'
    return bc


def merge_bc_columns(bc):
    """Input is idiosyncractic eflora bc data frame with scraped data (i.e., three columns holding content:
    content, morphology_content, morphology_content_misc. Returns a data frame with collapsed text,
    with the three content columns removed, replaced by a column named text. Due to string collapse, some elements
    may contain two whitespaces and nothing else."""
    bc[["content", "morphology_content", "morphology_content_misc", "species"]] \
        = bc[["content", "morphology_content", "morphology_content_misc", "species"]].fillna(value="")
    bc = bc[bc.classification.index.isna() == False]
    bc = bc[bc.classification.isna() == False]  # Drop columns with no classification
    assert bc.isna().any().sum() == 0, 'nan has not been removed'
    bc["text"] = bc["content"].map(str) + " " + bc["morphology_content"].map(str) + " " \
                 + bc["morphology_content_misc"].map(str)
    bc.drop(columns=["content", "morphology_content", "morphology_content_misc"], inplace=True)
    assert len(bc.columns) == 3, 'Reformatted data frame does not have three columns'
    return bc


def make_bc_data_frame(bc_filepath="data/external/eflora-bc-partial.csv", frac_to_sample=1, balance_categories=True):
    bc = read_bc_csv(bc_filepath)
    bc = merge_bc_columns(bc)
    bc.classification.cat.rename_categories({'keys': 'key', 'name': 'taxon_identification'}, inplace=True)
    bc_with_length = add_length_to_data_frame(bc)
    sampled_bc = sample_flora(bc_with_length, frac_to_sample, balance_categories)
    return sampled_bc
