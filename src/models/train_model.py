from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from src.data.make_fna import *
from src.data.make_bc import *
from src.data.make_budds import *
from src.features.build_features import *


fna = make_fna_data_frame(fna_filepath="data/external/fna_keys.csv", frac_to_sample=0.1, balance_categories=True)
bc = make_bc_data_frame(bc_filepath="data/external/efloratest-explode.csv",
                        frac_to_sample=1, balance_categories=True)
budds = make_budds_data_frame(budds_file_path="data/external/buddsfloraofcana00otta_djvu.xml", frac_to_sample=1,
                              balance_categories=True)
flora_data_frame = pd.concat([fna, bc, budds])

# Model Generation Using Multinomial Naive Bayes
tokenized_stop_words = prepare_stop_words(fna_stop_words=["unknown", "accepted", "synonym", "basionym"])

def run_model(text_counts, flora_data_frame):
    X_train, X_test, y_train, y_test = build_train_test_split(text_counts, flora_data_frame)
    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
    return predicted

# ==== DTM =====
dtm_text_counts = build_dtm_text_counts(flora_tokenizer, tokenized_stop_words, flora_data_frame)
predictions = run_model(dtm_text_counts, flora_data_frame)

# ==== TFIDF =====
tfidf_text_counts = build_tfidf_text_counts(flora_tokenizer, tokenized_stop_words, flora_data_frame)
predictions = run_model(tfidf_text_counts, flora_data_frame)

# other things to try: 1-gram vs. 2-gram

# fna_nb_classifier <- naiveBayes(fna_train, fna_train_labels, laplace = 1, CV = 10)  # laplace=1: dig down
# saveRDS(fna_nb_classifier, file = "models/budds_classifier.Rds")
# fna_nb_classifier <- readRDS(file = 'classifier.Rds')

# fna_test_pred_nb <- predict(fna_nb_classifier, newdata = fna_test)
# saveRDS(fna_test_pred_nb, file = "models/prediction.Rds")

# Missed = fna_test[which(fna_test_pred_nb != fna_test_labels), ]
# Succesful = fna_test[which(fna_test_pred_nb == fna_test_labels), ]

# saveRDS(fna_test, file = "data/interim/r_studio_fna_test.Rds")

# save.image("models/budds_classifier_data.RData")
