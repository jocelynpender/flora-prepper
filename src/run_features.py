import logging
import sys
import joblib
import pandas as pd

from features import build_dtm_text_counts, prepare_stop_words, flora_tokenizer


def main(flora_data_frame_path, dump_path):
    """This is my default feature build with no cleaning regime and no length features."""

    logger = logging.getLogger(__name__)
    logger.info('building features using provided data frame and dumping them at the specified path')

    flora_data_frame = pd.read_csv(flora_data_frame_path, index_col=0)
    tokenized_stop_words = prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                                 "basionym", "source",
                                                                 "note", "notes", "morphology", "fna_id"])
    custom_vec, dtm_text_counts = build_dtm_text_counts(flora_tokenizer, tokenized_stop_words,
                                                        flora_data_frame)
    file_names = [dump_path + file_name for file_name in ["custom_vec", "dtm_text_counts"]]
    joblib.dump(custom_vec, file_names[0])
    joblib.dump(dtm_text_counts, file_names[1])

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    flora_data_frame_path = sys.argv[1]
    dump_path = sys.argv[2]
    main(flora_data_frame_path, dump_path)
