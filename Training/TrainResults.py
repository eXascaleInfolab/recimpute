"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
TrainResults.py
@author: @chacungu
"""

import pandas as pd

class TrainResults:
    """
    Class which handles and stores results of different models' training on the same data over the same cross-validation splits.
    """


    # constructor
    
    def __init__(self, models):
        multiindex = pd.MultiIndex(names=['CV Split ID', 'Model Name'], levels=[[],[]], labels=[[],[]])
        columns = [
            'Accuracy', 'F1-Score', 'Precision', 'Recall', 'Hamming Loss',
            'Conf Matrix', 'Trained Pipeline',
        ]
        self.results = pd.DataFrame(index=multiindex, columns=columns)
        self.models = models


    # public methods

    def add_model_cv_split_result(self, model, trained_pipeline, scores, cm):
        # TODO
        pass

    def get_model_results(self, model_name):
        # TODO
        pass

    def save(self):
        # TODO
        pass