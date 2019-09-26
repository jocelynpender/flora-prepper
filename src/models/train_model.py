from src.features.build_features import *
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Model Generation Using Multinomial Naive Bayes

text_counts, X_train, X_test, y_train, y_test = build_fna_dtm()
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))

text_counts, X_train, X_test, y_train, y_test = build_fna_tfidf()
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))

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
