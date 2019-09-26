# also encode in the file the option to produce an tf/idf not just dtm

# tokenizer to remove unwanted elements from out data like symbols and numbers

#  https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.data.make_fna import *
from src.features.build_fna import *


def build_fna_dtm():
    fna = make_fna_data_frame(fna_filepath="data/external/fna_keys.csv", balance_categories=True)
    tokenized_stop_words = prepare_stop_words(fna_stop_words = ["unknown", "accepted", "synonym", "basionym"])

    custom_vec = CountVectorizer(lowercase=True, tokenizer=fna_tokenize, stop_words=tokenized_stop_words, ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(fna['text'])  # Build Document-Term-Matrix

    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, fna['classification'], test_size=0.3, random_state=1)

    return text_counts, X_train, X_test, y_train, y_test




def build_fna_tfidf():
    fna = make_fna_data_frame(fna_filepath="data/external/fna_keys.csv", balance_categories=True)
    tokenized_stop_words = prepare_stop_words(fna_stop_words = ["unknown", "accepted", "synonym", "basionym"])

    custom_vec = TfidfVectorizer(lowercase=True, tokenizer=fna_tokenize, stop_words=tokenized_stop_words, ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(fna['text'])  # Build Document-Term-Matrix

    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, fna['classification'], test_size=0.3, random_state=1)

    return text_counts, X_train, X_test, y_train, y_test

text_counts, X_train, X_test, y_train, y_test = build_fna_tfidf()


# Restrict features to frequent terms

# freq_terms <- findFreqTerms(fna_dtm_train, 5)
# saveRDS(freq_terms, file = "models/budds_freq_terms.Rds")

# fna_dtm_freq_terms_train <- fna_dtm_train[, freq_terms]
# fna_dtm_freq_terms_test <- fna_dtm_test[, freq_terms]

# fna_train <- apply(fna_dtm_freq_terms_train, 2, yes.no)
# fna_test <- apply(fna_dtm_freq_terms_test, 2, yes.no)
