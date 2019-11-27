#!/usr/bin/env python
# coding: utf-8

# # Predicting Budds text and reformatting its schema for CharaParser
# ## Workflow steps:
# 
# * Import text file by reading the lines
# * Push the lines into the model pipeline
# * add the classifications as XML nodes
# * reformat to input schema

# In[64]:


# Auto update of packages within the notebook
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys

# Import custom modelling code
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.predict_model import *
from src.features.build_length_features import locate_empty_strings


# ## Importing Budd's text

# In[13]:


budds_file_name = "/Users/jocelynpender/flora-prepper-proj/data/external/texts/budds_testing.txt"
budds_lines = open(budds_file_name).read().splitlines()


# In[ ]:


# if two key tags occur right after one another, merge the key tags into one key tag


# ## Import the vectorizer and create Budd's DTM

# In[20]:


custom_vec = joblib.load("../../models/custom_vec") # load the DTM vectorizer
test_dtm = custom_vec.transform(budds_lines) # fit the DTM to the new data


# ## Import the classifier and generate predictions

# In[22]:


# load the model and generate predictions
clf = joblib.load("../../models/classifier_model") # load the classifier
predicted = clf.predict(test_dtm) # predict the classification based on the DTM and the model
dtm_predictions_series = pd.Series(predicted)


# ## Reformat the results for easy viewing

# In[37]:


budds_lines = pd.Series(budds_lines)
budds_results = pd.concat([budds_lines, dtm_predictions_series], axis=1)
budds_results = budds_results[locate_empty_strings(budds_results[0])] # Remove rows with empty strings as text
budds_results.to_csv(path_or_buf="budds_results_to_examine.csv")
budds_results

