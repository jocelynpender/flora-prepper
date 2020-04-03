import os
import joblib
import pandas as pd


class StableFloraPrepper:
    """
    Run the stable release of the Flora Prepper model. You can access the stored location of the model
    files by exploring the StableFloraPrepper attributes.

    To leverage this class and run the stable Flora Prepper model in your Python project, add this to the top of your script:

    sys.path.insert(0, 'path/to/your/local/repository/clone/flora-prepper/src') # Add the project's source directory to your Python path
    from stable_model import StableFloraPrepper # import the stable model class

    flora_prepper = StableFloraPrepper() # initialize the stable model
    flora_canada_predictions = flora_prepper.generate_predictions(your_data) # run the model and generate predictions in your Python script

    """

    def __init__(self):
        models_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models/stable'))
        self.file_path = models_file_path
        self.custom_vec = models_file_path + "/custom_vec"
        self.classifier_model = models_file_path + "/classifier_model"

    def generate_predictions(self, data):
        """
        :param data: a list of sentences to be classified
        :return: predicted classifications of the sentences
        """
        custom_vec = joblib.load(self.custom_vec)
        clf = joblib.load(self.classifier_model)  # TODO update the most recent models with new budds data

        test_dtm = custom_vec.transform(data)
        predicted = clf.predict(test_dtm)
        dtm_predictions_series = pd.Series(predicted)
        return dtm_predictions_series

