#!/usr/bin/env python
# coding: utf-8

# Sept 30th-Oct 1st

# In[1]:


from src.data.make_fna import *
from src.data.make_bc import *
from src.data.make_budds import *
from src.features.build_features import *
from nltk.tokenize import word_tokenize
from src.visualization.visualize import *


# In[2]:


# Build data frames
fna = make_fna_data_frame(fna_filepath="../data/external/fna_keys.csv", frac_to_sample=0.1, balance_categories=True,
                          categories_to_keep=["key", "morphology", "taxon_identification", "distribution"])
bc = make_bc_data_frame(bc_filepath="../data/external/eflora-bc-full.csv",
                        frac_to_sample=1, balance_categories=True)
budds = make_budds_data_frame(budds_file_path="../data/raw/buddsfloraofcana00otta_djvu.xml", frac_to_sample=1,
                              balance_categories=True)
flora_data_frame = pd.concat([fna, bc, budds])


# Word cloud for all text data

# In[3]:


tokenized_stop_words = prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note", "notes"])  # Find a way to keep numbers and elipses!
text = " ".join(text_string for text_string in flora_data_frame.text)
visualize_words(text, tokenized_stop_words)


# Word clouds by classification

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


# Word clouds by flora source

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


# Key vs morphology UNCLEANED

# In[13]:


tokenized_key=word_tokenize(key)
fdist_key = FreqDist(tokenized_key)
fdist_key.plot(30,cumulative=False)
plt.show()

tokenized_morphology=word_tokenize(morphology)
fdist_morphology = FreqDist(tokenized_morphology)
fdist_morphology.plot(30,cumulative=False)
plt.show()


# Key vs morphology CLEANED

# In[14]:


word_cloud_key = WordCloud(stopwords=tokenized_stop_words, 
                      background_color="white", collocations=False).process_text(key)
fdist_key = FreqDist(word_cloud_key)
fdist_key.plot(30,cumulative=False)
plt.show()

word_cloud_morphology = WordCloud(stopwords=tokenized_stop_words, 
                      background_color="white", collocations=False).process_text(morphology)
fdist_morphology = FreqDist(word_cloud_morphology)
fdist_morphology.plot(30,cumulative=False)
plt.show()


# Oct. 2nd

# In[18]:


plt.figure(figsize=(18, 6), dpi=80)

processed_text = flora_tokenizer(bc_text) # Tokenize
processed_text = [word for word in processed_text if word.lower() not in tokenized_stop_words]  # Remove stop words
fdist = FreqDist(processed_text)

plt.subplot(1, 2, 1)
wordcloud = WordCloud(background_color="white").generate_from_frequencies(fdist)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

plt.subplot(1, 2, 2)
fdist.plot(30, cumulative=False)
plt.show()
print(fdist.most_common(30))


# Oct. 4th
# #### Are the habitat strings from bc and the distribution strings from fna similar enough to be merged?

# In[ ]:


sample_dist = flora_data_frame[flora_data_frame.classification == 'distribution'].text.iloc[:10]
print("Distribution text: ", sample_dist)

sample_hab = flora_data_frame[flora_data_frame.classification == 'habitat'].text.iloc[:10]
print("Habitat text: ", sample_hab)


# My conclusion is that we need to extract the habitat text that we have from fna and merge it with the bc strings.
# For the purposes of this poster. merge the distribution and habitat strings from FNA together for training!

# ### Oct 8
# ### What happens to the key and morphology classifications if we implement stricter word tokenization?

# In[ ]:


#key

word_cloud_key = WordCloud(stopwords=tokenized_stop_words, 
                      background_color="white", collocations=False).process_text(key)
fdist_key = FreqDist(word_cloud_key)
fdist_key.plot(30,cumulative=False)
plt.show()

# morphology
word_cloud_morphology = WordCloud(stopwords=tokenized_stop_words, 
                      background_color="white", collocations=False).process_text(morphology)
fdist_morphology = FreqDist(word_cloud_morphology)
fdist_morphology.plot(30,cumulative=False)
plt.show()

