from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np


def build_train_test_split(text_counts, data_frame):
    """
    Build the training and test sets with some default arguments.
    :param text_counts: The csr_matrix DTM or TFIDF object
    :param data_frame: The entire flora_data_frame with a classification column
    :return: the training/testing split objects
    """
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data_frame['classification'], test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test


def print_top10(vectorizer, clf, class_labels): # https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
    """Prints features with the highest coefficient values, per class

    :param vectorizer:
    :param clf:
    :param class_labels:
    :return:
    """
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))


def run_model(text_counts, flora_data_frame, feature_rank=False, custom_vec=None):
    """
    Builds training and testing set using flora_data_frame classification and runs model on text counts (
    features).
    :param text_counts:
    :param flora_data_frame:
    :param feature_rank:
    :param custom_vec:
    :return:
    """
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

    return clf, y_test, predicted


def zero_rule_algorithm_classification(train, test): # A baseline model. Compare performance of NB to this model
    """

    :param train:
    :param test:
    :return:
    """
    output_values = [row[-1] for row in train] # [-1] is useful when you don’t have the length of the container,
    # and want to reference a position relative to the last index without having to calculate the length.
    prediction = max(set(output_values), key=output_values.count) # returns the class value that has the highest
    # count of observed values in the list of class values observed in the training dataset.
    predicted = [prediction for i in range(len(test))]
    return predicted


# other things to try: 1-gram vs. 2-gram
# Test bigrams/unigrams
