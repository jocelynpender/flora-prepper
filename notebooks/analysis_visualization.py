#!/usr/bin/env python
# coding: utf-8

# # Flora Prepper Model Evaluation
# ## Initialize the environment

# In[30]:


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

from src.models.run_model import *
from src.visualization.visualize import *
import src.features as features


# ## Import and visualize the data
# ### Visualize counts of training datasets

# In[2]:


flora_data_frame = pd.read_csv("../data/processed/flora_data_frame.csv")
flora_data_frame['dataset_name'].value_counts().plot.bar()
plt.show()
flora_data_frame[['classification', 'dataset_name', 'text']] .groupby(['classification', 'dataset_name']).count().plot.bar()


# ### View a word cloud for all text data 
# Text is processed using the same custom (bare-bones) tokenizer and stopwords used to train the model. 
# 

# In[3]:


tokenized_stop_words = features.prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note", "notes", "morphology"])  # Find a way to keep numbers and elipses!
# morphology word is an artefact of the FNA xml key statements. 
# TODO Return to this and fix
text = " ".join(text_string for text_string in flora_data_frame.text)
visualize_words(text, tokenized_stop_words)


# #### What does the word cloud look like if we apply a strict preprocessing cleaning regime?

# In[31]:


tokenized_stop_words_clean = features.prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note", "notes", "morphology"], clean=True) 
visualize_words(text, tokenized_stop_words_clean, cleanup=True)


# ### Generate word clouds by classification.
# Are there any noticeable differences between the words used most frequently between the classifications?

# In[4]:


taxon_identification = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.classification == "taxon_identification"].text)
morphology = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.classification == "morphology"].text)
key = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.classification == "key"].text)
habitat = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.classification == "habitat"].text)


# Taxon identification

# In[5]:


visualize_words(taxon_identification, tokenized_stop_words)


# Morphology

# In[6]:


visualize_words(morphology, tokenized_stop_words)


# Keys

# In[7]:


visualize_words(key, tokenized_stop_words)


# Habitat

# In[8]:


visualize_words(habitat, tokenized_stop_words)


# ### Word clouds by flora source
# Are there differences between training sets in the most commonly used words?

# In[9]:


bc_text = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.dataset_name == 'bc'].text if text_string not in tokenized_stop_words)
budds_text = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.dataset_name == 'budds'].text if text_string not in tokenized_stop_words)
fna_text = " ".join(text_string for text_string in flora_data_frame[flora_data_frame.dataset_name == 'fna'].text if text_string not in tokenized_stop_words)


# BC

# In[10]:


visualize_words(bc_text, tokenized_stop_words)


# FNA

# In[11]:


visualize_words(fna_text, tokenized_stop_words)


# Budds

# In[12]:


visualize_words(budds_text, tokenized_stop_words)


# ### What happens to the key and morphology classifications if we implement stricter word tokenization?
# 
# Key

# In[13]:


word_cloud_key = WordCloud(stopwords=tokenized_stop_words, 
                      background_color="white", collocations=False).process_text(key)
fdist_key = FreqDist(word_cloud_key)
fdist_key.plot(30,cumulative=False)
plt.show()


# Morphology

# In[14]:


word_cloud_morphology = WordCloud(stopwords=tokenized_stop_words, 
                      background_color="white", collocations=False).process_text(morphology)
fdist_morphology = FreqDist(word_cloud_morphology)
fdist_morphology.plot(30,cumulative=False)
plt.show()


# ### Visualize distinctive words using tf-idf
# 

# In[15]:


custom_vec = TfidfVectorizer(lowercase=True, tokenizer=features.flora_tokenizer, stop_words=tokenized_stop_words, ngram_range=(1, 1))
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
dtm_text_counts = build_dtm_text_counts(features.flora_tokenizer, tokenized_stop_words, flora_data_frame)
dtm_X_test, dtm_predictions = run_model(dtm_text_counts, flora_data_frame)

# ==== TFIDF =====
tfidf_text_counts = build_tfidf_text_counts(features.flora_tokenizer, tokenized_stop_words, flora_data_frame)
tfidf_X_test, tfidf_predictions = run_model(tfidf_text_counts, flora_data_frame)


# #### View classified statements

# In[17]:


results = zip(dtm_X_test, dtm_predictions)
print(tuple(results)[:10])


# In[18]:


# TODO: View incorrectly classified statements

# Currently not working

#for item, labels in zip(X_test, predictions):
 #   print('%s => %s' % (item, ', '.join(flora_data_frame.classification[x] for x in labels)))


# ## Run a model based on text length

# In[19]:


# Process text, remove stopwords. Remove empty cells.
length_processed_flora_data_frame = features.process_length_in_place(flora_data_frame, tokenized_stop_words)

plot = length_processed_flora_data_frame['length'].hist(by=length_processed_flora_data_frame['classification'])
plt.show()


# It looks like discussion should be removed from the dataset. It is curiously short in length. This may be an artifact from the bc dataset.

# In[20]:


length_custom_vec = CountVectorizer(lowercase=True, tokenizer=features.flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
length_text_counts = length_custom_vec.fit_transform(length_processed_flora_data_frame['text'])

length_model_sparse = features.prepare_length_features(length_text_counts, length_custom_vec, length_processed_flora_data_frame)

X_test, predicted = run_model(length_model_sparse, length_processed_flora_data_frame)


# In[21]:


#fig,ax = plt.subplots(figsize=(5,5))
#plt.boxplot(df_train_1.phrase_len)
#plt.show()


# To do plots:
# classification coloured by source

# ## Run a model with only the most frequently occurring words

# In[22]:


all_text = " ".join(text_string for text_string in flora_data_frame.text)
all_text = features.flora_tokenizer(all_text)
top_words_text = features.find_most_frequent_words(all_text, threshold=2000)
top_words_flora_data_frame = features.filter_data_frame_top_words(flora_data_frame, top_words_text, tokenized_stop_words)
top_words_flora_data_frame


# In[23]:


all_text_custom_vec = CountVectorizer(lowercase=True, tokenizer=features.flora_tokenizer, stop_words=tokenized_stop_words,
                                 ngram_range=(1, 1))
all_text_counts = all_text_custom_vec.fit_transform(top_words_flora_data_frame['text'])
X_test, predicted = run_model(all_text_counts, top_words_flora_data_frame)

