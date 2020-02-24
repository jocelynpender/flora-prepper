import argparse
import joblib
import pandas as pd
import logging

import scipy

from models.build_model import run_model


def main(flora_data_frame_path, dtm_text_counts_path, model_path):
    """
    Build the classifier model using the training dataset, the DTM text counts (i.e., the features) and save it for
    future use.
    :param flora_data_frame_path: Training dataset
    :param dtm_text_counts_path: The DTM built in the make_features.py step
    :param model_path: Where to store the resulting classifier model
    :return:
    """

    logger = logging.getLogger(__name__)
    logger.info('making model from provided feature set')

    # Read in training dataset and text counts DTM
    flora_data_frame = pd.read_csv(flora_data_frame_path, index_col=0)
    # custom_vec = joblib.load(custom_vec_path)
    dtm_text_counts = joblib.load(dtm_text_counts_path)
    assert type(dtm_text_counts) == scipy.sparse.csr.csr_matrix, 'DTM text counts should be a csr matrix'

    # Run model build and save the file
    clf, dtm_y_test, dtm_predictions = run_model(dtm_text_counts, flora_data_frame)
    file_name = model_path + "classifier_model"
    joblib.dump(clf, file_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description='Build Naive Bayes classifier model')
    parser.add_argument('--train_file_path', type=str, help='path to the training dataset')
    parser.add_argument('--dtm_file_path', type=str, default="models/dtm_text_counts",
                        help='a list of stop words')
    parser.add_argument('--model_save_path', type=str, default="models/", help='path to save model')
    args = parser.parse_args()

    main(args.train_file_path, args.dtm_file_path, args.model_save_path)