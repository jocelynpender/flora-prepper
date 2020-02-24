#!/usr/bin/env python
# coding: utf-8

# # Predicting Budds text and reformatting its schema for CharaParser
# 
# * Import text file by reading the lines
# * Push the lines into the model pipeline
# * Add the classifications as XML nodes
# * Reformat to input schema
# * Break up data into XML documents

# In[14]:


# Auto update of packages within the notebook
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys
import numpy as np

# Import custom modelling code
module_path = os.path.abspath(os.path.join('../../../'))
module_path = os.path.abspath(os.path.join('../../../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)

from src.make_predict import *
from src.features.build_length_features import locate_empty_strings
from src.transformers import classification_xml


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

# In[ ]:


# Workflow execution
budds_results = pd.read_csv("flora_commons_workflow/budds_results_to_examine_rekey.csv", index_col=0)
budds_results.reclassification[0] = budds_results.classification[0]  # Fix first item
budds_strings_classification = merge_classification_blocks(budds_results, 2)

# https://github.com/biosemantics/schemas/blob/master/semanticMarkupInput.xsd
prematter = '<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<bio:treatment '             'xmlns:bio=\"http://www.github.com/biosemantics\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" '             'xsi:schemaLocation=\"http://www.github.com/biosemantics http://www.w3.org/2001/XMLSchema-instance\"> '

open_tags = {'taxon_identification': prematter + '\n' + '<taxon_identification status="ACCEPTED">',
             'morphology': '<description type="morphology">',
             'habitat': '<description type="habitat">',
             'key': '<key>'
             }

close_tags = {'taxon_identification': '</taxon_identification>',
              'morphology': '</description>',
              'habitat': '</description>',
              'key': '</key>'
              }

write_budds_docs = write_documents(budds_strings_classification, open_tags, close_tags, prematter)


# In[ ]:




