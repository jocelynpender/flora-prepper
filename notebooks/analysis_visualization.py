#!/usr/bin/env python
# coding: utf-8

# # Flora Prepper Model Evaluation
# ## Initialize the environment

# In[1]:


from src.data.make_fna import *
from src.data.make_bc import *
from src.data.make_budds import *
from src.models.train_model import *
from src.visualization.visualize import *


# ## Import the data

# In[2]:


fna = make_fna_data_frame(fna_filepath="../data/external/fna_keys.csv", frac_to_sample=0.1, balance_categories=True,
                          categories_to_keep=["key", "morphology", "taxon_identification", "distribution"])
bc = make_bc_data_frame(bc_filepath="../data/external/eflora-bc-partial.csv",
                        frac_to_sample=1, balance_categories=True)
budds = make_budds_data_frame(budds_file_path="../data/external/buddsfloraofcana00otta_djvu.xml", frac_to_sample=1,
                              balance_categories=True)
flora_data_frame = pd.concat([fna, bc, budds])


# View a word cloud for all text data. Text is processed using the same tokenizer and stopwords used to train the model. 
# 

# In[3]:


tokenized_stop_words = prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note", "notes"])  # Find a way to keep numbers and elipses!
text = " ".join(text_string for text_string in flora_data_frame.text)
visualize_words(text, tokenized_stop_words)


# Word clouds by classification.
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


# Word clouds by flora source. Are there differences between training sets in the most commonly used words?

# In[9]:


bc_text = " ".join(text_string for text_string in bc.text if text_string not in tokenized_stop_words)
budds_text = " ".join(text_string for text_string in budds.text if text_string not in tokenized_stop_words)
fna_text = " ".join(text_string for text_string in fna.text if text_string not in tokenized_stop_words)


# BC

# In[10]:


visualize_words(bc_text, tokenized_stop_words)


# FNA

# In[11]:


visualize_words(fna_text, tokenized_stop_words)


# Budds

# In[12]:


visualize_words(budds_text, tokenized_stop_words)


# What happens to the key and morphology classifications if we implement stricter word tokenization?
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


# Run a DTM based model and a TFIDF based model and review accuracy

# In[15]:


# ==== DTM =====
dtm_text_counts = build_dtm_text_counts(flora_tokenizer, tokenized_stop_words, flora_data_frame)
predictions = run_model(dtm_text_counts, flora_data_frame)

# ==== TFIDF =====
tfidf_text_counts = build_tfidf_text_counts(flora_tokenizer, tokenized_stop_words, flora_data_frame)
predictions = run_model(tfidf_text_counts, flora_data_frame)

