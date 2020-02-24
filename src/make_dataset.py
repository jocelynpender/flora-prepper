# -*- coding: utf-8 -*-
import argparse
import logging
import pandas as pd

from data.make_bc import make_bc_data_frame
from data.make_budds import make_budds_data_frame
from data.make_flora import make_flora_data_frame


def append_training_datasets_to_flora_data_frame(flora_data_frame, data_to_append, dataset_name_to_append):
    # Concatenate all the component datasets together
    data_to_append = data_to_append.assign(dataset_name = dataset_name_to_append)
    flora_data_frame = flora_data_frame.append(data_to_append, sort=False)
    return flora_data_frame


def shuffle_drop_reset_flora_data_frame(flora_data_frame):
    # Do some last cleaning steps
    flora_data_frame = flora_data_frame.sample(frac=1)  # Shuffle the dataset in place
    flora_data_frame = flora_data_frame.drop(flora_data_frame[flora_data_frame.text == "  "].index)  # drop weirdo
    # rows with " " text
    flora_data_frame = flora_data_frame.reset_index()
    flora_data_frame = flora_data_frame.rename(columns={"level_0": "row_id"}) # rename old row id after resetting index
    return flora_data_frame


def main(output_filepath, fna_filepath=None, bc_filepath=None, budds_file_path=None, fm_file_path=None):
    """
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

    :param fna_filepath: path to the Flora of North America raw dataset
    :param bc_filepath: path to the Illustrated Flora of BC raw dataset
    :param budds_file_path: path to the Budds raw xml file
    :param fm_file_path: path to Flora of Manitoba raw dataset
    :param output_filepath: a filename to store the final flora training dataset
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    flora_data_frame = pd.DataFrame()

    # Build component datasets
    if fna_filepath:
        fna = make_flora_data_frame(fna_filepath, frac_to_sample=1, balance_categories=True,
                              categories_to_keep=["key", "morphology", "taxon_identification",
                                                  "distribution", "habitat"])
        flora_data_frame = append_training_datasets_to_flora_data_frame(flora_data_frame, fna, "fna")
    if bc_filepath:
        bc = make_bc_data_frame(bc_filepath, frac_to_sample=1, balance_categories=True,
                            categories_to_keep=["key", "morphology", "taxon_identification", "habitat"])
        flora_data_frame = append_training_datasets_to_flora_data_frame(flora_data_frame, bc, "bc")
    if budds_file_path:
        budds = make_budds_data_frame(budds_file_path, frac_to_sample=1, balance_categories=True)
        flora_data_frame = append_training_datasets_to_flora_data_frame(flora_data_frame, budds, "budds")
    if fm_file_path:
        fm = make_flora_data_frame(fm_file_path, frac_to_sample=1, balance_categories=False,
                               categories_to_keep=["key", "morphology", "taxon_identification", "distribution"],
                               rename_habitat=False)
        flora_data_frame = append_training_datasets_to_flora_data_frame(flora_data_frame, fm, "fm")

    flora_data_frame = shuffle_drop_reset_flora_data_frame(flora_data_frame)
    flora_data_frame.to_csv(output_filepath)
    return flora_data_frame

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description='Build flora training dataset from scratch')
    parser.add_argument('--fna_filepath', type=str, default="data/raw/fna_with_habitat.csv",
                        help='path to the Flora of North America raw dataset')
    parser.add_argument('--bc_filepath', type=str, default="data/raw/eflora_bc.csv",
                        help='path to the Illustrated Flora of BC raw dataset')
    parser.add_argument('--budds_file_path', type=str, default="data/raw/buddsfloraofcana00otta_djvu.xml",
                        help='path to the Budds raw xml file')
    parser.add_argument('--fm_file_path', type=str, default="data/raw/fm.csv",
                        help='path to the raw Flora of Manitoba xml file')
    parser.add_argument('--output_filepath', type=str, default="data/processed/flora_data_frame.csv",
                        help='a filename to store the final flora training dataset')
    args = parser.parse_args()

    main(args.output_filepath, fna_filepath=args.fna_filepath, bc_filepath=args.bc_filepath,
         budds_file_path=args.budds_file_path, fm_file_path=args.fm_file_path)
