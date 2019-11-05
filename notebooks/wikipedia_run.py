#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Auto update of packages within the notebook
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Packages required for model building and analysis
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


# In[2]:


# Import model training data
flora_data_frame = pd.read_csv("../data/processed/flora_data_frame_full.csv", index_col=0)
train_indices = list(range(0, flora_data_frame.shape[0]))


# In[3]:


# Import of Wikipedia dataset
wiki = pd.read_csv("../data/raw/cirsium_arvense_wikipedia.csv", index_col=None)
test_indices = list(range(flora_data_frame.shape[0] + 1, flora_data_frame.shape[0] + wiki.shape[0]))
flora_data_frame = pd.concat([flora_data_frame, wiki], ignore_index=True)
flora_data_frame


# In[4]:


# Customize stop words for model
tokenized_stop_words = features.prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note", "notes", "morphology", "fna_id"])
# Build DTM
custom_vec, dtm_text_counts = build_dtm_text_counts(features.flora_tokenizer, tokenized_stop_words, flora_data_frame)
dtm_text_counts.toarray()
print(dtm_text_counts.shape)
print(flora_data_frame.shape)


# In[5]:


# Prepare data for the model
X_train = dtm_text_counts[train_indices]
y_train = flora_data_frame.iloc[train_indices].classification
X_test = dtm_text_counts[test_indices]
y_test = flora_data_frame.iloc[test_indices].classification


# In[6]:


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




