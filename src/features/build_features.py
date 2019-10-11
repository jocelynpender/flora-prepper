# also encode in the file the option to produce an tf/idf not just dtm
# tokenizer to remove unwanted elements from out data like symbols and numbers
#  https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


def build_dtm_text_counts(flora_tokenizer, tokenized_stop_words, data_frame):
    custom_vec = CountVectorizer(lowercase=True, tokenizer=flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(data_frame['text'])  # Build Document-Term-Matrix
    return custom_vec, text_counts


def build_tfidf_text_counts(flora_tokenizer, tokenized_stop_words, data_frame):
    custom_vec = TfidfVectorizer(lowercase=True, tokenizer=flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(data_frame['text'])  # Build TF-IDF Matrix
    return custom_vec, text_counts


def build_train_test_split(text_counts, data_frame):
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data_frame['classification'], test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test