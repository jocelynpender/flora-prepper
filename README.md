flora-prepper
==============================

Prepare flora documents from OCR to XML for CharaParser.

Data Credits
Flora of North America. The Flora of North America Association has graciously provided Flora of North America data to the Integrated Flora of Canada project. Flora Prepper is part of this project. Copyright on the data is held by the Flora of North America Association for all volumes except Volumes 24 and 25 (Poaceae). Copyright for Volumes 24 and 25 is held by Utah State University. Data was accessed in 2019.
Budd’s Flora of the Canadian Prairie Provinces. Copyright is held by the Government of Canada. This Flora was accessed through the Biodiversity Heritage Library in 2019: https://www.biodiversitylibrary.org/item/114250
Illustrated Flora of British Columbia. Copyright is held by the Province of British Columbia. Provisional permission to begin investigations of the use of this data in the Integrated Flora of Canada project was obtained by members of the Illustrated Flora of BC project. Data cannot and will not be shared, except within the scope of data integration experiments, such as the above.

References
Hamann, T. et al. 2014. Detailed mark-up of semi-monographic legacy taxonomic works using FlorML. Taxon 63 (2): 377-393. https://doi.org/10.12705/632.11
Sautter, G. et al. 2002 Semi-automated XML markup of biosystematic legacy literature with the GoldenGate editor. Biocomputing 2007. https://doi.org/10.1142/9789812772435_0037


Discussion points not included on the poster:
The accuracy of the model is unusually high. When the proportion of Flora of North Am. and B.C. data was reduced by dataset trimming, accuracy decreased. Is overfitting occurring?
Keys and morphology text blocks are very similar. How can classification of these two categories be improved?
Habitat text separates well
This model relies too heavily on artefacts of the training dataset (rank words) to classify taxon identification strings. Instead, they could be annotated using other tools, like gnfinder (github.com/gnames/gnfinder)


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
