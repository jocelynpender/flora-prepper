"""
Generating predictions.
"""
import argparse
import logging
import joblib
import pandas as pd


def main(test_file_path, col_name, results_path, model_path, custom_vec_path):
    """Load previously constructed DTM and classifier model.
    Generate predictions from an input csv file of strings taken from a flora.

    :param test_file_path: The file to run predictions on. The format of this file will be a csv file, where one column
    contains strings to be classified. This pipeline is designed to be run on data within a Pandas workflow and paradigm.
    :param col_name: The column of the csv containing the strings to be classified.
    :param model_path: The path to the model to be used in this prediction exercise.
    :param custom_vec_path: The path to the vectorizer that builds the DTM and matches the input data to the DTM (i.e.,
    feature building).
    :param results_path: The path where the results.csv output file will be stored.
    :return: results.csv: a csv file that matches the input, but with an additional column named 'predictions' containing
    model predictions.
    """

    logger = logging.getLogger(__name__)
    logger.info('building predictions from model')

    # build dtm with the input data_frame
    test_data = pd.read_csv(test_file_path)
    custom_vec = joblib.load(custom_vec_path) # load the DTM vectorizer
    test_dtm = custom_vec.transform(test_data[col_name]) # fit the DTM to the new data

    # load the model and generate predictions
    clf = joblib.load(model_path) # load the classifier
    predicted = clf.predict(test_dtm) # predict the classification based on the DTM and the model
    dtm_predictions_series = pd.Series(predicted)

    # write the results to a csv
    results = pd.concat([test_data, dtm_predictions_series], axis=1)
    results.rename(columns={0: 'predictions'}, inplace=True)
    #results = results.set_index('index')
    print(list(results.columns))
    results_file_name = results_path + "results.csv"
    results.to_csv(path_or_buf=results_file_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description='Use a Naive Bayes text mining classifier to predict the '
                                                 'classification of strings (morphology, habitat, name, keys)')
    parser.add_argument('--test_file_path', type=str, help='path to the file you would like classified')
    parser.add_argument('--test_file_column_name', type=str, default="text",
                        help='name of the column that contains strings to be classified')
    parser.add_argument('--results_path', type=str, default="reports/",
                        help='path where the results file will be stored')
    parser.add_argument('--model_path', type=str, default="models/classifier_model",
                        help='path to the model to be used')
    parser.add_argument('--vectorizer_path', type=str, default="models/custom_vec",
                        help='path to the vectorizer to be used to construct a DTM')
    args = parser.parse_args()

    # Validate arguments

    assert args.test_file_path, "Argument --test_file_path is required for generating predictions"

    main(args.test_file_path, args.test_file_column_name, args.results_path, args.model_path, args.vectorizer_path)
    # test_file_path, col_name, model_path, custom_vec_path, results_path)