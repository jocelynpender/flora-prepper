# also encode in the file the option to produce an tf/idf not just dtm
# tokenizer to remove unwanted elements from out data like symbols and numbers
#  https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split


def flora_tokenizer(input_string):
    """Apply custom tokenization to fna strings"""
    assert type(input_string) == str, 'Text column not string format'
    input_string = word_tokenize(input_string)
    assert type(input_string) == list, 'Tokens not returned in text column as list'
    return input_string


def build_nltk_stop_words(stop_words):
    """Takes a set of stop words and add nltk's english words to it"""
    import nltk
    nltk.download('stopwords')
    nltk_stop_words = set(stopwords.words('english'))
    assert len(nltk_stop_words) > 0, 'NLTK words not loaded'
    stop_words += nltk_stop_words
    return stop_words


def prepare_stop_words(custom_stop_words, include_nltk_stop_words=True):
    """Tokenize the stop words"""
    assert type(custom_stop_words) == list, 'Stop words not provided as a list'
    if include_nltk_stop_words:
        custom_stop_words = build_nltk_stop_words(custom_stop_words)
    tokenized_stop_words = []
    for word in custom_stop_words:
        tokenized_stop_words.extend(word_tokenize(word))
    tokenized_stop_words = frozenset(tokenized_stop_words)
    return tokenized_stop_words


def build_dtm_text_counts(flora_tokenizer, tokenized_stop_words, data_frame):
    custom_vec = CountVectorizer(lowercase=True, tokenizer=flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(data_frame['text'])  # Build Document-Term-Matrix
    return text_counts


def build_tfidf_text_counts(flora_tokenizer, tokenized_stop_words, data_frame):
    custom_vec = TfidfVectorizer(lowercase=True, tokenizer=flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
    text_counts = custom_vec.fit_transform(data_frame['text'])  # Build Document-Term-Matrix
    return text_counts


def build_train_test_split(text_counts, data_frame):
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data_frame['classification'], test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test

# Restrict features to frequent terms

# freq_terms <- findFreqTerms(fna_dtm_train, 5)
# saveRDS(freq_terms, file = "models/budds_freq_terms.Rds")

# fna_dtm_freq_terms_train <- fna_dtm_train[, freq_terms]
# fna_dtm_freq_terms_test <- fna_dtm_test[, freq_terms]

# fna_train <- apply(fna_dtm_freq_terms_train, 2, yes.no)
# fna_test <- apply(fna_dtm_freq_terms_test, 2, yes.no)
