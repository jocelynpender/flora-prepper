import lxml
from lxml import etree
import pandas as pd
from src.data.make_helper_functions import *


def return_text_series(tree, path):
    """:param:
        A parsed etree from an XML file and the desired node path to extract text from
    :return:
        A nicely formatted pandas Series containing text data extracted from the node."""
    assert len(path) > 0, "Path not valid"
    text_list = [element.text for element in tree.findall(path)] # Extract text from all XML nodes
    assert type(text_list) == list, "List comprehension not run"
    text_series = pd.Series(text_list) # Convert the list into a series for downstream data frame concatenation
    assert type(text_series) == pd.core.series.Series, "List comprehension not run"
    return text_series


def extract_classification_text(tree):
    """Run extraction on all classification types that exist in the Budds flora XML file.
    :param:
        A parsed etree from an XML file.
    :return:
        Four pandas Series based on the four classifications."""
    taxon_identification = return_text_series(tree, "//taxon_identification")
    key = return_text_series(tree, "//key")
    morphology = return_text_series(tree, "//description")
    return taxon_identification, key, morphology


def make_budds_data_frame(budds_file_path, frac_to_sample=1, balance_categories=True):
    """Parse the Budds Flora XML file, extract classification text data, and concatenate it all together in a pandas DataFrame.
    Run sampling and balancing if desired.
    :param:
        File path to XML file.
    :return:
        Data frame with text data parsed, length added, trimming and balancing performed."""
    tree = etree.parse(budds_file_path)
    assert type(tree) == lxml.etree._ElementTree, 'Tree not parsed properly'
    taxon_identification, key, morphology = extract_classification_text(tree)
    classifications = ["taxon_identification", "key", "morphology"]
    budds = pd.concat([taxon_identification, key, morphology], keys=classifications,
                      names=["classification", "row"])
    budds = budds.reset_index()  # Moves names from index into columns in a new data frame
    assert type(budds) == pd.core.frame.DataFrame, "Budds not converted to DataFrame"
    budds.columns = ["classification", "row", "text"]
    budds_with_length = add_length_to_data_frame(budds)
    sampled_budds = sample_flora(budds_with_length, frac_to_sample, balance_categories)
    return sampled_budds
