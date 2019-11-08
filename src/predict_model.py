"""
Generating predictions.
"""

import logging
import sys
import joblib

import pandas as pd

from features import prepare_stop_words


def main(test_file_path, col_name, model_path, custom_vec_path, results_path):
    """

    :param test_file_path:
    :param clf:
    :param results_path:
    :return:
    """

    logger = logging.getLogger(__name__)
    logger.info('building predictions from model')

    test_data = pd.read_csv(test_file_path)  # build dtm with the input data_frame

    tokenized_stop_words = prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                                 "basionym", "source",
                                                                 "note", "notes", "morphology", "fna_id"])
    custom_vec = joblib.load(custom_vec_path)

    X_test = custom_vec.transform(test_data[col_name])
    # X_test =

    clf = joblib.load(model_path)

    predicted = clf.predict(X_test)
    dtm_predictions_series = pd.Series(predicted)

    results = pd.concat([test_data, dtm_predictions_series], axis=1)
    results.rename(columns={0: 'predictions'}, inplace=True)
    #results = results.set_index('index')

    results_file_name = results_path + "results.csv"
    results.to_csv(path_or_buf = results_file_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    test_file_path = sys.argv[1]
    col_name = sys.argv[2]
    model_path = sys.argv[3]
    custom_vec_path =  sys.argv[4]
    results_path = sys.argv[5]
    main(test_file_path, col_name, model_path, custom_vec_path, results_path)
