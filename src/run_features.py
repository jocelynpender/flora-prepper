import argparse
import logging
import joblib
import pandas as pd

from features import build_dtm_text_counts, prepare_stop_words, flora_tokenizer


def main(flora_data_frame_path, dump_path, custom_stop_words_path):
    """This is my default feature build with no cleaning regime and no length features.

    :param flora_data_frame_path: The training dataset to build features and the model from
    :param dump_path: Where to save the DTM vectorizer and the classifier model
    :param custom_stop_words_path: Where to find custom stop words to be used in the model
    :return:
    """

    logger = logging.getLogger(__name__)
    logger.info('building features using provided data frame and dumping them at the specified path')

    # Import the training dataset
    flora_data_frame = pd.read_csv(flora_data_frame_path, index_col=0)

    # Import custom stop words and build tokenized stopwords objects
    custom_stop_words_list = open(custom_stop_words_path).read().splitlines()
    tokenized_stop_words = prepare_stop_words(custom_stop_words=custom_stop_words_list)

    # Run feature build
    custom_vec, dtm_text_counts = build_dtm_text_counts(flora_tokenizer, tokenized_stop_words,
                                                        flora_data_frame)

    # Save the DTM vectorizer and classifier model for later usage
    file_names = [dump_path + file_name for file_name in ["custom_vec", "dtm_text_counts"]]
    joblib.dump(custom_vec, file_names[0])
    joblib.dump(dtm_text_counts, file_names[1])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description='Build features for classifier model')
    parser.add_argument('--train_file_path', type=str, help='path to the training dataset')
    parser.add_argument('--features_save_path', type=str, default="models/", help='path to save features')
    parser.add_argument('--custom_stop_words_path', type=str, default="models/stop_words.txt",
                        help='a list of stop words')
    args = parser.parse_args()

    main(args.train_file_path, args.features_save_path, args.custom_stop_words_path)