# -*- coding: utf-8 -*-
import logging
from data.make_bc import make_bc_data_frame
from data.make_budds import make_budds_data_frame
from data.make_fna import make_fna_data_frame
import pandas as pd


def main(output_filepath="data/processed/flora_data_frame.csv"):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    fna = make_fna_data_frame(fna_filepath="data/external/fna_with_habitat.csv", frac_to_sample=0.1,
                              balance_categories=True,
                              categories_to_keep=["key", "morphology", "taxon_identification",
                                                  "distribution", "discussion", "habitat"])
    bc = make_bc_data_frame(bc_filepath="data/external/eflora-bc-full_no-id.csv",
                            frac_to_sample=0.13, balance_categories=True,
                            categories_to_keep=["key", "morphology", "taxon_identification", "habitat", "discussion"])
    budds = make_budds_data_frame(budds_file_path="data/external/buddsfloraofcana00otta_djvu.xml", frac_to_sample=1,
                                  balance_categories=True)
    flora_data_frame = pd.concat([fna, bc, budds], keys=['fna', 'bc', 'budds'], names=['dataset_name', 'row_id'])
    flora_data_frame = flora_data_frame.sample(frac=1)  # Shuffle the dataset in place
    flora_data_frame = flora_data_frame.reset_index()

    flora_data_frame.to_csv(path_or_buf=output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()



