import os
import sys

import nltk

from features.build_stop_words import prepare_stop_words
from features.process_text import flora_tokenizer, find_most_frequent_words

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data.make_fna import *
from src.data.make_bc import *
from src.data.make_budds import *
from src.models.build_model import *
from src.visualization.visualize import *
from src.features.build_features import *
import numpy as np

fna = make_fna_data_frame(fna_filepath="../data/raw/fna_with_habitat.csv", frac_to_sample=0.1, balance_categories=True,
                          categories_to_keep=["key", "morphology", "taxon_identification",
                                              "distribution", "discussion", "habitat"])
bc = make_bc_data_frame(bc_filepath="../data/raw/eflora_bc.csv",
                        frac_to_sample=0.13, balance_categories=True,
                        categories_to_keep=["key", "morphology", "taxon_identification", "habitat", "discussion"])
budds = make_budds_data_frame(budds_file_path="../data/raw/buddsfloraofcana00otta_djvu.xml", frac_to_sample=1,
                              balance_categories=True)
flora_data_frame = pd.concat([fna, bc, budds], keys=['fna', 'bc', 'budds'], names=['dataset_name', 'row_id'])

flora_data_frame = flora_data_frame.sample(frac=1)  # Shuffle the dataset in place
flora_data_frame = flora_data_frame.reset_index()

tokenized_stop_words = prepare_stop_words(custom_stop_words=["unknown", "accepted", "synonym",
                                                             "basionym", "source",
                                                             "note",
                                                             "notes"])  # Find a way to keep numbers and elipses!

# ==== DTM =====
dtm_text_counts = build_dtm_text_counts(flora_tokenizer, tokenized_stop_words, flora_data_frame)
clf, dtm_X_test, dtm_predictions = run_model(dtm_text_counts, flora_data_frame)

# ==== TFIDF =====
tfidf_text_counts = build_tfidf_text_counts(flora_tokenizer, tokenized_stop_words, flora_data_frame)
clf, tfidf_X_test, tfidf_predictions = run_model(tfidf_text_counts, flora_data_frame)

taxon_identification = " ".join(
    text_string for text_string in flora_data_frame[flora_data_frame.classification == "taxon_identification"].text)

taxon_identification = process_text(taxon_identification, tokenized_stop_words)
word_features = find_most_frequent_words(taxon_identification)


# ==== SCRAP ====


# tfidf_text_counts = build_tfidf_text_counts(flora_tokenizer, tokenized_stop_words, flora_data_frame)
# print(tfidf_text_counts)

custom_vec = TfidfVectorizer(lowercase=True, tokenizer=flora_tokenizer, stop_words=tokenized_stop_words,
                             ngram_range=(1, 1))
text_counts = custom_vec.fit_transform(flora_data_frame['text'])  # Build TF-IDF Matrix

scores = zip(custom_vec.get_feature_names(), np.asarray(text_counts.sum(axis=0)).ravel())
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
for item in sorted_scores:
    print("{0:50} Score: {1}".format(item[0], item[1]))

test = list(text_counts.toarray())
print(test)

df = pd.DataFrame(text_counts.toarray())

df3 = pd.DataFrame(sorted_scores, columns=['word', 'score']).iloc[:50]
df3.plot.bar(x='word', y='score')
plt.show()
