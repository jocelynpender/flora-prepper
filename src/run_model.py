import joblib
import pandas as pd
import logging
import sys

from models.build_model import run_model


def main(flora_data_frame_path, dtm_text_counts_path, model_path):
    """

    :param flora_data_frame_path:
    :param custom_vec_path:
    :param dtm_text_counts_path:
    :param model_path:
    :return:
    """

    logger = logging.getLogger(__name__)
    logger.info('making model from provided feature set')

    flora_data_frame = pd.read_csv(flora_data_frame_path, index_col=0)
    # custom_vec = joblib.load(custom_vec_path)
    dtm_text_counts = joblib.load(dtm_text_counts_path)
    clf, dtm_y_test, dtm_predictions = run_model(dtm_text_counts, flora_data_frame)
    file_name = model_path + "classifier_model"
    joblib.dump(clf, file_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    flora_data_frame_path = sys.argv[1]
    dtm_text_counts_path = sys.argv[2]
    model_path = sys.argv[3]
    main(flora_data_frame_path, dtm_text_counts_path, model_path)
