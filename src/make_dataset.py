# -*- coding: utf-8 -*-
import logging
import sys
import pandas as pd

from data.make_bc import make_bc_data_frame
from data.make_budds import make_budds_data_frame
from data.make_fna import make_fna_data_frame


def main(fna_filepath, bc_filepath, budds_file_path, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    fna = make_fna_data_frame(fna_filepath, frac_to_sample=1, balance_categories=True,
                              categories_to_keep=["key", "morphology", "taxon_identification",
                                                  "distribution", "habitat"])
    bc = make_bc_data_frame(bc_filepath, frac_to_sample=1, balance_categories=True,
                            categories_to_keep=["key", "morphology", "taxon_identification", "habitat"])
    budds = make_budds_data_frame(budds_file_path, frac_to_sample=1, balance_categories=True)
    flora_data_frame = pd.concat([fna, bc, budds], keys=['fna', 'bc', 'budds'], names=['dataset_name', 'row_id'])
    flora_data_frame = flora_data_frame.sample(frac=1)  # Shuffle the dataset in place
    flora_data_frame = flora_data_frame.drop(flora_data_frame[flora_data_frame.text == "  "].index) # drop weirdo rows with " " text
    flora_data_frame = flora_data_frame.reset_index()
    flora_data_frame.to_csv(path_or_buf=output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    fna_filepath = sys.argv[1]
    bc_filepath = sys.argv[2]
    budds_file_path = sys.argv[3]
    output_filepath = sys.argv[4]
    main(fna_filepath, bc_filepath, budds_file_path, output_filepath)
