#!/usr/bin/env python
# coding: utf-8

# # Flora Prepper Model Evaluation
# ## Initialize the environment

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import numpy as np
import pandas as pd

# Import custom modelling code
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.models.build_model import *
from src.visualization.visualize import *
import src.features as features


# ## Import and visualize the data
# ### Visualize counts of training datasets

# In[2]:


flora_data_frame = pd.read_csv("../data/processed/flora_data_frame_full.csv", index_col=0)
flora_data_frame['dataset_name'].value_counts().plot.bar()
plt.show()
flora_data_frame[['classification', 'dataset_name', 'text']] .groupby(['classification', 'dataset_name']).count().plot.bar()


# ### View a word cloud for all text data 
# Text is processed using the same custom (bare-bones) tokenizer and stopwords used to train the model. 
# 

# In[3]:


tokenized_stop_words = features.prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note", "notes", "morphology", "fna_id"])  # Find a way to keep numbers and elipses!
# morphology word is an artefact of the FNA xml key statements. 
# TODO Return to this and fix
text = " ".join(text_string for text_string in flora_data_frame.text)
visualize_words(text, tokenized_stop_words)


# #### What does the word cloud look like if we apply a strict preprocessing cleaning regime?

# In[4]:


tokenized_stop_words_clean = features.prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note", "notes", "morphology", "fna_id"], clean=True) 
visualize_words(text, tokenized_stop_words_clean, cleanup=True)


# ### Generate word clouds by classification.
# Are there any noticeable differences between the words used most frequently between the classifications?

# In[5]:


taxon_identification = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.classification == "taxon_identification"].text)
morphology = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.classification == "morphology"].text)
key = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.classification == "key"].text)
habitat = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.classification == "habitat"].text)


# Taxon identification

# In[75]:


visualize_words(taxon_identification, tokenized_stop_words, color="purple")
visualize_words(taxon_identification, tokenized_stop_words_clean, cleanup=True, color="purple")


# Morphology

# In[7]:


visualize_words(morphology, tokenized_stop_words, color="red")
visualize_words(morphology, tokenized_stop_words_clean, cleanup=True, color="red")


# Keys

# In[8]:


visualize_words(key, tokenized_stop_words, color="yellow")
visualize_words(key, tokenized_stop_words_clean, cleanup=True, color="yellow")


# Habitat

# In[74]:


visualize_words(habitat, tokenized_stop_words, color="blue")
visualize_words(habitat, tokenized_stop_words_clean, cleanup=True, color="blue")


# ### Word clouds by flora source
# Are there differences between training sets in the most commonly used words?

# In[10]:


bc_text = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.dataset_name == 'bc'].text if text_string not in tokenized_stop_words)
budds_text = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.dataset_name == 'budds'].text if text_string not in tokenized_stop_words)
fna_text = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.dataset_name == 'fna'].text if text_string not in tokenized_stop_words)


# BC

# In[11]:


visualize_words(bc_text, tokenized_stop_words)
visualize_words(bc_text, tokenized_stop_words_clean, cleanup=True)


# FNA

# In[12]:


visualize_words(fna_text, tokenized_stop_words)
visualize_words(fna_text, tokenized_stop_words_clean, cleanup=True)


# Budds

# In[13]:


visualize_words(budds_text, tokenized_stop_words)
visualize_words(budds_text, tokenized_stop_words_clean, cleanup=True)


# ### Visualize distinctive words using tf-idf
# 

# In[14]:


custom_vec = TfidfVectorizer(lowercase=True, tokenizer=features.flora_tokenizer, stop_words=tokenized_stop_words, ngram_range=(1, 1))
text_counts = custom_vec.fit_transform(flora_data_frame['text'])  # Build TF-IDF Matrix

scores = zip(custom_vec.get_feature_names(), np.asarray(text_counts.sum(axis=0)).ravel())
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
#for item in sorted_scores:
 #   print("{0:10} Score: {1}".format(item[0], item[1]))
    
sorted_scores_df = pd.DataFrame(sorted_scores, columns=['word', 'score']).iloc[:50]
sorted_scores_df.plot.bar(x='word', y='score')
plt.show()


# Distinctive words with new cleaning regime

# In[15]:


custom_vec = TfidfVectorizer(lowercase=True, tokenizer=features.flora_tokenizer_clean, stop_words=tokenized_stop_words_clean, ngram_range=(1, 1))
text_counts = custom_vec.fit_transform(flora_data_frame['text'])  # Build TF-IDF Matrix

scores = zip(custom_vec.get_feature_names(), np.asarray(text_counts.sum(axis=0)).ravel())
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
#for item in sorted_scores:
 #   print("{0:10} Score: {1}".format(item[0], item[1]))
    
sorted_scores_df = pd.DataFrame(sorted_scores, columns=['word', 'score']).iloc[:50]
sorted_scores_df.plot.bar(x='word', y='score')
plt.show()


# ## Run a DTM based model and a TFIDF based model and review accuracy

# In[16]:


# ==== DTM =====
custom_vec, dtm_text_counts = build_dtm_text_counts(features.flora_tokenizer, tokenized_stop_words, flora_data_frame)
dtm_y_test, dtm_predictions = run_model(dtm_text_counts, flora_data_frame, feature_rank=True, custom_vec)

# ==== TFIDF =====
custom_vec, tfidf_text_counts = build_tfidf_text_counts(features.flora_tokenizer, tokenized_stop_words, flora_data_frame)
tfidf_y_test, tfidf_predictions = run_model(tfidf_text_counts, flora_data_frame, feature_rank=True, custom_vec)


# #### View classified statements

# In[17]:


dtm_y_test_df = pd.DataFrame(dtm_y_test).reset_index()
dtm_predictions_series = pd.Series(dtm_predictions)
results = pd.concat([dtm_y_test_df, dtm_predictions_series], axis=1)
results.rename(columns={0: 'predictions'}, inplace=True)
results = results.set_index('index')
results_flora_data_frame = pd.concat([results, flora_data_frame], axis=1)
results_flora_data_frame


# In[18]:


incorrect = results[results.classification != results.predictions]
incorrect_data_frame = results_flora_data_frame.iloc[incorrect.index]
incorrect_data_frame.to_csv(path_or_buf = "../reports/incorrect_dtm_clean.csv")
incorrect_data_frame


# ### Run with strict cleaning regime

# In[19]:


# ==== DTM =====
custom_vec, dtm_text_counts = build_dtm_text_counts(features.flora_tokenizer_clean, tokenized_stop_words_clean, flora_data_frame)
dtm_y_test, dtm_predictions = run_model(dtm_text_counts, flora_data_frame, feature_rank=True, custom_vec)

# ==== TFIDF =====
custom_vec, tfidf_text_counts = build_tfidf_text_counts(features.flora_tokenizer_clean, tokenized_stop_words_clean, flora_data_frame)
tfidf_y_test, tfidf_predictions = run_model(tfidf_text_counts, flora_data_frame, feature_rank=True, custom_vec)


# In[20]:


dtm_y_test_df = pd.DataFrame(dtm_y_test).reset_index()
dtm_predictions_series = pd.Series(dtm_predictions)
results = pd.concat([dtm_y_test_df, dtm_predictions_series], axis=1)
results.rename(columns={0: 'predictions'}, inplace=True)
results = results.set_index('index')
results_flora_data_frame = pd.concat([results, flora_data_frame], axis=1)
results_flora_data_frame


# In[21]:


incorrect = results[results.classification != results.predictions]
incorrect_data_frame = results_flora_data_frame.iloc[incorrect.index]
incorrect_data_frame.to_csv(path_or_buf = "../reports/incorrect_dtm_dirty.csv")
incorrect_data_frame


# ## Run a model based on text length

# In[22]:


# Process text, remove stopwords. Remove empty cells.
length_processed_flora_data_frame = features.process_length_in_place(flora_data_frame, tokenized_stop_words)


# In[72]:


length_processed_flora_data_frame.to_csv(path_or_buf = "../data/interim/length_processed_flora_data_frame.csv")


# In[71]:


classifications = length_processed_flora_data_frame.groupby(by=length_processed_flora_data_frame['classification'])

#bins = np.linspace(-10, 10, 100)

for group in classifications.groups:
#   print(group)
    group_df = length_processed_flora_data_frame[length_processed_flora_data_frame.classification == group]
    group_text_length = group_df['length']
    plt.hist(group_text_length, label=group)
#file_name = group + "_text_length.png"
#print(file_name)
#plt.savefig(file_name)
plt.xlim(0, 15000)
plt.ylim(0, 4000)
plt.legend(loc='upper right')
plt.show()

    
    # TO DO 
    
    # redo these as overlapping hist with four colours!!!


# It looks like discussion should be removed from the dataset. It is curiously short in length. This may be an artifact from the bc dataset.

# In[27]:


length_custom_vec = CountVectorizer(lowercase=True, tokenizer=features.flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
length_text_counts = length_custom_vec.fit_transform(length_processed_flora_data_frame['text'])

length_model_sparse = features.prepare_length_features(length_text_counts, length_custom_vec, length_processed_flora_data_frame)

X_test, predicted = run_model(length_model_sparse, length_processed_flora_data_frame, feature_rank=False)


# To do plots:
# classification coloured by source

# ## Run a model with only the most frequently occurring words

# In[28]:


all_text = " ".join(text_string for text_string in flora_data_frame.text)
all_text = features.flora_tokenizer(all_text)
top_words_text = features.find_most_frequent_words(all_text, threshold=2000)
top_words_flora_data_frame = features.filter_data_frame_top_words(flora_data_frame, top_words_text, tokenized_stop_words)
top_words_flora_data_frame


# In[30]:


all_text_custom_vec = CountVectorizer(lowercase=True, tokenizer=features.flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
all_text_counts = all_text_custom_vec.fit_transform(top_words_flora_data_frame['text'])
X_test, predicted = run_model(all_text_counts, top_words_flora_data_frame, feature_rank=False)


# ## Do models work well on previously unseen floras?

# In[42]:


custom_vec, dtm_text_counts = build_dtm_text_counts(features.flora_tokenizer, tokenized_stop_words, flora_data_frame)


# In[45]:


dtm_text_counts.toarray()
print(dtm_text_counts.shape)
print(flora_data_frame.shape)


# In[59]:


train_indices = flora_data_frame[(flora_data_frame['dataset_name'] =="bc") | (flora_data_frame['dataset_name'] == "fna")].index
X_train = dtm_text_counts[train_indices]
y_train = flora_data_frame.iloc[train_indices].classification
print(X_train.shape)
print(len(y_train))
test_indices = flora_data_frame[flora_data_frame['dataset_name'] =="budds"].index
print(len(test_indices))
X_test = dtm_text_counts[test_indices]
y_test = flora_data_frame.iloc[test_indices].classification


# In[60]:


clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted))
class_labels = clf.classes_
print_top10(custom_vec, clf, class_labels)


# In[63]:


dtm_y_test_df = pd.DataFrame(y_test).reset_index()
dtm_predictions_series = pd.Series(predicted)
results = pd.concat([dtm_y_test_df, dtm_predictions_series], axis=1)
results.rename(columns={0: 'predictions'}, inplace=True)
results = results.set_index('index')
results_flora_data_frame = pd.concat([results, flora_data_frame], axis=1, join='inner')
results_flora_data_frame


# ### Running a baseline model!

# In[ ]:


baseprediction = zero_rule_algorithm_classification(line_features, labels)  # get predictions for baseline

    if baseprediction[0] == 0:
        baseline = (len(baseprediction) - np.count_nonzero(labels)) / (len(baseprediction))
    else:
        baseline = (len(baseprediction) - len(np.where(labels == 0))) / (len(baseprediction))

    print(f"Baseline score: {baseline}")

