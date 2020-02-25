#!/usr/bin/env python
# coding: utf-8

# In[19]:


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
module_path = os.path.abspath(os.path.join('../../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)

print(sys.path)

from src.models.build_model import *
from src.visualization.visualize import *
from src.features.build_features import *
from src.data.make_wikipedia import *
from src.features.build_stop_words import *
from src.features.process_text import *

# pip install git+https://github.com/lucasdnd/Wikipedia.git --upgrade


# In[3]:


# Import model training data
flora_data_frame = pd.read_csv("../../data/processed/flora_data_frame.csv", index_col=0)
train_indices = list(range(0, flora_data_frame.shape[0]))


# In[4]:


# Import of Wikipedia dataset
wiki = pd.read_csv("../../data/processed/combined_wikidata.csv", index_col=None)
test_indices = list(range(flora_data_frame.shape[0] + 1, flora_data_frame.shape[0] + wiki.shape[0]))
flora_data_frame = pd.concat([flora_data_frame, wiki], ignore_index=True)

# Get rid of key classification
flora_data_frame.classification[flora_data_frame.classification == "key"] = "morphology"


# In[11]:


# Customize stop words for model
tokenized_stop_words = prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note", "notes", "morphology", "fna_id"])
# Build DTM
custom_vec, dtm_text_counts = build_dtm_text_counts(flora_tokenizer, tokenized_stop_words, flora_data_frame)
dtm_text_counts.toarray()


# In[12]:


# Prepare data for the model
X_train = dtm_text_counts[train_indices]
y_train = flora_data_frame.iloc[train_indices].classification
X_test = dtm_text_counts[test_indices]
y_test = flora_data_frame.iloc[test_indices].classification


# In[13]:


clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
dtm_y_test_df = pd.DataFrame(y_test).reset_index()
dtm_predictions_series = pd.Series(predicted)
results = pd.concat([dtm_y_test_df, dtm_predictions_series], axis=1)
results.rename(columns={0: 'predictions'}, inplace=True)
results = results.set_index('index')
results_flora_data_frame = pd.concat([results, flora_data_frame], axis=1, join='inner')
results_flora_data_frame.to_csv(path_or_buf = "../../reports/csv/wikidata_results.csv")


# In[17]:


X_test


# In[ ]:




