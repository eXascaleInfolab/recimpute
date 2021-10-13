"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
TrainResults.py
@author: @chacungu
"""

import datetime
from joblib import dump
from os.path import normpath as normp
import pandas as pd
import pickle
import random as rdm
import sys
import time

from Utils.Utils import Utils

class TrainResults:
    """
    Class which handles and stores results of different models' training on the same data over the same cross-validation splits.
    """

    RESULTS_DIR = normp('./Training/Results/')

    # create necessary directories if not there yet
    Utils.create_dirs_if_not_exist([RESULTS_DIR])


    # constructor
    
    def __init__(self, models):
        run_id = datetime.datetime.fromtimestamp(time.time()).strftime('%d%m_%H%M') 
        run_id = run_id + '_' + str(rdm.randint(0, sys.maxsize))[:5]
        self.id = run_id

        multiindex = pd.MultiIndex(names=['CV Split ID', 'Model'], levels=[[],[]], codes=[[],[]])
        columns = [
            'Accuracy', 'F1-Score', 'Precision', 'Recall', 'Hamming Loss',
            'Conf Matrix', 'Trained Pipeline',
        ]
        self.results = pd.DataFrame(index=multiindex, columns=columns)
        self.models = models


    # public methods

    def add_model_cv_split_result(self, split_id, model, trained_pipeline, scores, cm):
        """
        Stores the model's training results after a cross-validation split.

        Keyword arguments:
        split_id -- cross-validation split id
        model -- RecommendationModel instance whose pipeline has been trained
        trained_pipeline -- trained pipeline
        scores -- dict of scores measured during the model's evaluation
        cm -- tuple containing the confusion matrix's Matplotlib Figure, its Matplotlib Axes, and its values (numpy nd array)

        Return: -
        """
        assert model in self.models

        self.results.loc[(split_id, model),:] = {
            'Accuracy': scores['Accuracy'],
            'F1-Score': scores['F1-Score'],
            'Precision': scores['Precision'],
            'Recall': scores['Recall'],
            'Hamming Loss': scores['Hamming Loss'],

            'Conf Matrix': cm,
            'Trained Pipeline': trained_pipeline,
        }

    def get_model_results(self, model):
        """
        Returns all training results for the given model (DataFrame with one row per cross-validation split).

        Keyword arguments:
        model -- RecommendationModel for which the training results should be returned

        Return: 
        Pandas DataFrame containing the cross-validation results of the model. One row per cross-validation split.
        Columns: CV Split ID (index), Accuracy, F1-Score, Precision, Recall, Hamming Loss, Conf Matrix, Trained Pipeline
        """
        return self.results.xs(model, level='Model', drop_level=True)
        
    def get_avg_model_results(self, model):
        """
        Returns the training results' average for the given model.

        Keyword arguments:
        model -- RecommendationModel for which the training results' average should be returned

        Return: 
        Pandas Series containing the average cross-validation results of the model. 
        Columns: Accuracy, F1-Score, Precision, Recall, Hamming Loss.
        """
        return self.get_model_results(model).loc[:, ~self.results.columns.isin(['Conf Matrix', 'Trained Pipeline'])].mean()

    def save(self):
        """
        Saves this TrainResult's instance to disk.

        Keyword arguments: -

        Return: -
        """
        with open(TrainResults.get_filename(self.id), 'wb') as f_out:
            pickle.dump(self, f_out, protocol=pickle.HIGHEST_PROTOCOL)

        # TODO save information file w/ data sets, features extracters, labeler, true labeler and their params, 
        # the models, their params ranges, the gridsearch results


    # static methods

    def load(id):
        """
        Loads a TrainResults instance from disk.

        Keyword arguments: 
        id -- id of the TrainResults instance to load

        Return: 
        TrainResults instance
        """
        with open(TrainResults.get_filename(id), 'rb') as f_in:
            return pickle.load(f_in)
    
    def get_filename(id):
        """
        Returns the filename of a TrainResults instance from its id.

        Keyword arguments: 
        id -- id of the TrainResults instance

        Return: 
        Filename of a TrainResults instance
        """
        return normp(TrainResults.RESULTS_DIR + f'/{id}.p')