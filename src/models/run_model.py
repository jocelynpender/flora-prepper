from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from src.features.build_features import *
from sklearn.metrics import classification_report
import numpy as np


def print_top10(vectorizer, clf, class_labels): # https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))


def run_model(text_counts, flora_data_frame, feature_rank=False, custom_vec=None):
    """Builds training and testing set using flora_data_frame classification and runs model on text counts (
    features). """
    X_train, X_test, y_train, y_test = build_train_test_split(text_counts, flora_data_frame)
    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
    #  print("MultinomialNB Precision:", metrics.average_precision_score(y_test, predicted))
    # print("MultinomialNB F1:", metrics.f1_score(y_test, predicted))
    print(classification_report(y_test, predicted))
    class_labels = clf.classes_
    if feature_rank:
        print_top10(custom_vec, clf, class_labels)

    return y_test, predicted

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

# OCT 1st
# WORK with morphology, keys and habitat/distribution text.

# Test bigrams/unigrams
