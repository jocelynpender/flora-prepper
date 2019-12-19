#!/usr/bin/env python
# coding: utf-8

# # Predicting Budds text and reformatting its schema for CharaParser
# ## LOG NOTEBOOK
# ### Nov. 27th 2019
# #### TODO: Build out workflow
# ##### Workflow steps:
# 
# * Import text file by reading the lines
# * Push the lines into the model pipeline
# * Add the classifications as XML nodes
# * Reformat to input schema
# * Break up data into XML documents

# In[3]:


# Auto update of packages within the notebook
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import numpy as np

# Import custom modelling code
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.predict_model import *
from src.features.build_length_features import locate_empty_strings


# ##### Importing Budd's text

# In[13]:


budds_file_name = "/Users/jocelynpender/flora-prepper-proj/data/external/texts/budds_testing.txt"
budds_lines = open(budds_file_name).read().splitlines()


# In[ ]:


# TODO: if two key tags occur right after one another, merge the key tags into one key tag


# ##### Import the vectorizer and create Budd's DTM

# In[20]:


custom_vec = joblib.load("../../models/custom_vec") # load the DTM vectorizer
test_dtm = custom_vec.transform(budds_lines) # fit the DTM to the new data


# ##### Import the classifier and generate predictions

# In[22]:


# load the model and generate predictions
clf = joblib.load("../../models/classifier_model") # load the classifier
predicted = clf.predict(test_dtm) # predict the classification based on the DTM and the model
dtm_predictions_series = pd.Series(predicted)


# ##### Reformat the results for easy viewing

# In[37]:


budds_lines = pd.Series(budds_lines)
budds_results = pd.concat([budds_lines, dtm_predictions_series], axis=1)
budds_results = budds_results[locate_empty_strings(budds_results[0])] # Remove rows with empty strings as text
budds_results.columns = ["text", "classification"] # Set column names
budds_results.to_csv(path_or_buf="budds_results_to_examine.csv")
budds_results


# ### Dec. 17th 2019
# ##### Find cells that are not keys, and that are surrounded by keys on either side!

# In[85]:


budds_results = pd.read_csv("budds_results_to_examine.csv", index_col=0)
budds_results.columns = ["text", "classification"]
budds_results['reclassification'] = 'blank' # Add a new columsn to modify
budds_results.reset_index(inplace = True, drop = True) # Index was not accurate anymore
budds_results


# In[86]:


for index, row in budds_results.iterrows():
    if index > 0 and index < len(budds_results):
        if budds_results.iloc[index-1].classification == 'key' and budds_results.iloc[index+1].classification == 'key' and row.classification != 'key':
            row.reclassification = 'key'
        else:
            row.reclassification = row.classification


# In[4]:


# budds_results.to_csv(path_or_buf="budds_results_to_examine_rekey.csv")
budds_results = pd.read_csv("budds_results_to_examine_rekey.csv", index_col=0)
budds_results


# ##### Add XML to classified tags

# In[5]:


# schema tags for budds classification:
# TODO: find schema document from Github

budds_results.reclassification[0] = budds_results.classification[0] # Fix first item


# In[6]:


# https://stackoverflow.com/questions/14358567/finding-consecutive-segments-in-a-pandas-data-frame
index_matrix = budds_results.reset_index().groupby('reclassification')['index'].apply(np.array) # Find sequences of classifications
#index_matrix['key']

budds_results['block'] = (budds_results.reclassification.shift(1) != budds_results.reclassification).astype(int).cumsum()
budds_results


# In[8]:


# Take the block, paste the text all together, and wrap it in the appropriate XML tag
# Build a dictionary of tag

# <taxon_identification status="ACCEPTED"></taxon_identification>
# <description type="morphology"></description>
# <description type="habitat"></description>
# <key></key>
open_tags = {'taxon_identification': '<taxon_identification status="ACCEPTED">',
             'morphology': '<description type="morphology">',
             'habitat': '<description type="habitat">',
             'key': '<key>'
            }
close_tags = {'taxon_identification': '</taxon_identification>',
              'morphology': '</description>',
              'habitat': '</description>',
              'key': '</key>'
    
}


# In[30]:


group_blocks = budds_results.groupby('block')
runs = group_blocks['text'].apply(np.array)
runs


# ### Dec. 18th 2019

# In[31]:


#'\n'.join(runs.iloc[0])
strings = runs.apply(lambda x: '\n'.join(x))
strings


# In[32]:


group_blocks['reclassification'].take([0])


# In[33]:


index_matrix


# In[ ]:




