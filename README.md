flora-prepper
==============================
## Details
This project:
(1) Cleans training datasets from three sources (Flora of North America, Budd's Flora of the Prairie Provinces, Illustrated Flora of British Columbia)
(2) Trains a Naive Bayes model to predict the classification of strings based on a document-term matrix

The goals are to:
(1) Prepare flora documents from OCR for use with CharaParser/markup with schemas.

## Read the poster
https://github.com/jocelynpender/flora-prepper/blob/master/reports/TDWG2019%20Poster%20PenderJ.pdf

### Data credits
The training data is currently not included in this repository. However, I would like to credit the data providers nonetheless.
#### Flora of North America
The Flora of North America Association has graciously provided Flora of North America data to the Integrated Flora of Canada project. Flora Prepper is part of this project. Copyright on the data is held by the Flora of North America Association for all volumes except Volumes 24 and 25 (Poaceae). Copyright for Volumes 24 and 25 is held by Utah State University. Data was accessed in 2019.
#### Budd’s Flora of the Canadian Prairie Provinces
Copyright is held by the Government of Canada. This Flora was accessed through the Biodiversity Heritage Library in 2019: https://www.biodiversitylibrary.org/item/114250
#### Illustrated Flora of British Columbia
Copyright is held by the Province of British Columbia. Provisional permission to begin investigations of the use of this data in the Integrated Flora of Canada project was obtained by members of the Illustrated Flora of BC project. Data cannot and will not be shared, except within the scope of data integration experiments, such as the above.

### References
Hamann, T. et al. 2014. Detailed mark-up of semi-monographic legacy taxonomic works using FlorML. Taxon 63 (2): 377-393. https://doi.org/10.12705/632.11
Sautter, G. et al. 2002 Semi-automated XML markup of biosystematic legacy literature with the GoldenGate editor. Biocomputing 2007. https://doi.org/10.1142/9789812772435_0037


### Discussion points not included on the poster:
The accuracy of the model is unusually high. When the proportion of Flora of North Am. and B.C. data was reduced by dataset trimming, accuracy decreased. Is overfitting occurring?

Keys and morphology text blocks are very similar. How can classification of these two categories be improved?
Habitat text separates well

This model relies too heavily on artefacts of the training dataset (rank words) to classify taxon identification strings. Instead, they could be annotated using other tools, like gnfinder (github.com/gnames/gnfinder)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
