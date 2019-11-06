#!/usr/bin/env python
# coding: utf-8

# In[113]:


# Auto update of packages within the notebook
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Packages required for model building and analysis
import os
import sys
import numpy as np
import pandas as pd
import wikipedia

# Import custom modelling code
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.models.run_model import *
from src.visualization.visualize import *
import src.features as features
from src.data.make_wikipedia import *

# pip install git+https://github.com/lucasdnd/Wikipedia.git --upgrade


# In[2]:


# Import model training data
flora_data_frame = pd.read_csv("../data/processed/flora_data_frame_full.csv", index_col=0)
train_indices = list(range(0, flora_data_frame.shape[0]))


# In[89]:


species_list = list(flora_data_frame.species.unique())
species_list = [species for species in species_list if str(species) != 'nan'] # Remove nan species names


# In[117]:


batches = [species_list[i: i+10] for i in range(0, len(species_list), 10)]
for num, batch in enumerate(batches):
    print(num)
    make_wiki_data_frame(batch)


# In[4]:


# Import of Wikipedia dataset
# wiki = pd.read_csv("../data/raw/cirsium_arvense_wikipedia.csv", index_col=None)
wiki = wiki_data
test_indices = list(range(flora_data_frame.shape[0] + 1, flora_data_frame.shape[0] + wiki.shape[0]))
flora_data_frame = pd.concat([flora_data_frame, wiki], ignore_index=True)
flora_data_frame


# In[20]:


# Get rid of key classification
flora_data_frame.classification[flora_data_frame.classification == "key"] = "morphology"
flora_data_frame


# In[5]:


# Customize stop words for model
tokenized_stop_words = features.prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note", "notes", "morphology", "fna_id"])
# Build DTM
custom_vec, dtm_text_counts = build_dtm_text_counts(features.flora_tokenizer, tokenized_stop_words, flora_data_frame)
dtm_text_counts.toarray()


# In[6]:


# Prepare data for the model
X_train = dtm_text_counts[train_indices]
y_train = flora_data_frame.iloc[train_indices].classification
X_test = dtm_text_counts[test_indices]
y_test = flora_data_frame.iloc[test_indices].classification


# In[7]:


clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
dtm_y_test_df = pd.DataFrame(y_test).reset_index()
dtm_predictions_series = pd.Series(predicted)
results = pd.concat([dtm_y_test_df, dtm_predictions_series], axis=1)
results.rename(columns={0: 'predictions'}, inplace=True)
results = results.set_index('index')
results_flora_data_frame = pd.concat([results, flora_data_frame], axis=1, join='inner')
results_flora_data_frame


# In[ ]:




