import lxml
from lxml import etree
import pandas as pd
from src.data.make_helper_functions import *


def return_text_series(tree, path):
    """Using the parsed etree and the node path, return a nicely formatted pandas Series containing text data extracted
    from the node."""
    assert len(path) > 0, "Path not valid"
    text_list = [element.text for element in tree.findall(path)] # Extract text from all XML nodes
    assert type(text_list) == list, "List comprehension not run"
    text_series = pd.Series(text_list) # Convert the list into a series for downstream dataframe concatenation
    assert type(text_series) == pd.core.series.Series, "List comprehension not run"
    return text_series


def extract_classification_text(tree):
    """Run extraction on all classification types and return four pandas Series"""
    taxon_identification = return_text_series(tree, "//taxon_identification")
    key = return_text_series(tree, "//key")
    morphology = return_text_series(tree, "//description")
    return taxon_identification, key, morphology


def make_budds_data_frame(budds_file_path, frac_to_sample=1, balance_categories=True):
    """Parse the XML file, extract classification text data, and concatenate it all together in a pandas DataFrame.
    Run sampling and balancing if desired."""
    tree = etree.parse(budds_file_path)
    assert type(tree) == lxml.etree._ElementTree, 'Tree not parsed properly'

    taxon_identification, key, morphology = extract_classification_text(tree)
    classifications = ["taxon_identification", "key", "morphology"]
    budds = pd.concat([taxon_identification, key, morphology], keys=classifications,
                      names=["classification", "row"])
    budds = budds.reset_index()  # Moves names from index into columns in a new dataframe
    assert type(budds) == pd.core.frame.DataFrame, "Budds not converted to DataFrame"
    budds.columns = ["classification", "row", "text"]

    sampled_budds = sample_flora(budds, frac_to_sample, balance_categories)
    return sampled_budds
