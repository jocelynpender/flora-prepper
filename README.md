![flora-prepper](reports/figures/poster_title.png)

==============================

This project:
1. Cleans training datasets from three floras (Flora of North America, Budd's Flora of the Prairie Provinces, Illustrated Flora of British Columbia)
2. Builds a multinomial Naive Bayes model to predict the classification of flora strings based on a document-term matrix (i.e., habitat, morphology, name, keys)

The goal is to:
1. Prepare flora documents for more fine-grained parsing and data integration by annotating data types ("classifications")

## Read the poster for details
[Poster](reports/TDWG2019%20Poster%20PenderJ.pdf)

## Getting started

### Prerequisites

* Python 3.7

See `requirements.txt`. It has been generated with `pip freeze > requirements.txt`.

#### The project was built within a conda environment.
See `flora-prepper.yml`. It has been generated with `conda env export > flora-prepper.yml`.

You can build your own using:
`conda env create -f flora-prepper.yml`

#### (Optional) Wikipedia library for Wikipedia page scraping

Installing the Wikipedia library may require manual intervention. Section titles are not properly extracted using the latest release of the Wikipedia Python library. Therefore, to download your Wikipedia library, use

Use `pip install git+https://github.com/lucasdnd/Wikipedia.git`
This issue has been flagged and resolved via https://stackoverflow.com/questions/34869597/wikipedia-api-for-python/35122688#35122688

### Run the model

First, test your environment using:
`make requirements`

To deploy the latest version of the model, you'll need:
* A test dataset file, in CSV format, e.g., `test_dataset.csv`
* The name of the column containing text, e.g., `dataset_column_text`

The run this one-liner in your terminal:
`python3 src/predict_model.py test_dataset.csv dataset_column_text models/classifier_model models/custom_vec reports/`

#### (Optional) Run full data pipeline

If you want to build the model from scratch, I've created some make commands that run the default dataset build, feature builds and model training. 

Most of the parameters I've decided to use for my default model are hard-coded as custom function parameters. You'll have to consult the code to see what is being run.

* `make data`: This builds the training dataset from raw to processed. 
* `make features`: This command constructs the requisite vectorizer and document-term-matrix to build the Naive Bayes model.
* `make model`: The model is built using the vectorizer and document-term-matrix from above.
* `make predict`: Runs a model prediction on Wikipedia page strings.

## Project organization

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
    ├── docs               <- WIP: stub sphinx docs.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Each notebook is accompanied by an up-to-date `.py` and 
    |                          `.html` version for better report reading and report version control.
    |                          The `log` notebook was used for logging data and model
    │                         explorations. 
    |                          The `analysis_visualization_log` notebook was used for more
    │                         full-fledged model explorations.
    |                          The `analysis_visualization` notebook is the canonical report containing
    |                          all model development and assessment. 
    |                         The `wikipedia_run` notebook was used to assess model performance on 
    |                          species Wikipedia pages.
    │                           
    │
    ├── reports            <- Generated analysis as PDF, pptx, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── csv            <- Generated csv for model performance reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── flora-prepper.yml   <- The requirements file for reproducing the analysis environment with conda,
    │                           e.g. generated with `conda env export > flora-prepper.yml`.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate data from raw (not included in repository)
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── run_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- WIP: tox file with settings for running tox; see tox.testrun.org.


--------


## Versioning

## License

[License](LICENSE)

## People first

### Collaborations
If you have a use case for this project, and would like to contribute or collaborate, please contact me at pender.jocelyn@gmail.com

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

### Discussion points not included on the poster
The accuracy of the model is unusually high. When the proportion of Flora of North Am. and B.C. data was reduced by dataset trimming, accuracy decreased. Is overfitting occurring?

Keys and morphology text blocks are very similar. How can classification of these two categories be improved?
Habitat text separates well

This model relies too heavily on artefacts of the training dataset (rank words) to classify taxon identification strings. Instead, they could be annotated using other tools, like gnfinder (github.com/gnames/gnfinder)

### Acknowledgements

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
