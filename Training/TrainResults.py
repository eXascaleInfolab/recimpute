"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
TrainResults.py
@author: @chacungu
"""

import datetime
from joblib import dump as j_dump, load as j_load
import os
from os.path import normpath as normp
import pandas as pd
from pickle import dump as p_dump, load as p_load
import numpy as np
import random as rdm
import sys
import time
import warnings
import zipfile

from Training.RecommendationModel import RecommendationModel
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
            'Conf Matrix',
        ]
        self.results = pd.DataFrame(index=multiindex, columns=columns)
        for col in 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'Hamming Loss':
            self.results[col] = self.results[col].astype(np.float64)
        self.models = models


    # public methods

    def add_model_cv_split_result(self, split_id, model, scores, cm):
        """
        Stores the model's training results after a cross-validation split.

        Keyword arguments:
        split_id -- cross-validation split id
        model -- RecommendationModel instance whose pipeline has been trained
        scores -- dict of scores measured during the model's evaluation
        cm -- tuple containing the confusion matrix's Matplotlib Figure, its Matplotlib Axes, and its values (numpy nd array)

        Return: -
        """
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        assert model in self.models

        key = (split_id, model)
        values = {
            'Accuracy': scores['Accuracy'],
            'F1-Score': scores['F1-Score'],
            'Precision': scores['Precision'],
            'Recall': scores['Recall'],
            'Hamming Loss': scores['Hamming Loss'],

            'Conf Matrix': cm,
        }
        self.results.loc[key,:] = values

    def get_model_results(self, model):
        """
        Returns all training results for the given model (DataFrame with one row per cross-validation split).

        Keyword arguments:
        model -- RecommendationModel for which the training results should be returned

        Return: 
        Pandas DataFrame containing the cross-validation results of the model. One row per cross-validation split.
        Columns: CV Split ID (index), Accuracy, F1-Score, Precision, Recall, Hamming Loss, Conf Matrix
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
        return self.get_model_results(model).loc[:, ~self.results.columns.isin(['Conf Matrix'])].mean()

    def get_model_best_split(self, model, metric='F1-Score'):
        """
        Returns the results of the model's best cross-validation split.

        Keyword arguments:
        model -- RecommendationModel for which the training results should be returned
        metric -- name of the score column to consider when sorting the results

        Return: 
        Pandas Series containing the average cross-validation results of the model. 
        Columns: Accuracy, F1-Score, Precision, Recall, Hamming Loss, Conf Matrix
        """
        return self.results.loc[(self.get_model_results(model)[metric].idxmax(), model), :]

    def save(self, training_set):
        """
        Saves this TrainResult's instance to disk.

        Keyword arguments: 
        training_set -- TrainingSet instance used for the models' training

        Return: -
        """
        archive_filename, pickle_filename, info_filename = TrainResults._get_filenames(self.id)

        # create an info file
        with open(info_filename, 'w') as f_out:
            f_out.write('# Information about those results:')
            f_out.write('\n## Data sets used:')
            f_out.write( ''.join('\n- ' + str(d) for d in training_set.datasets) )
            f_out.write('\n## Test set\'s IDs:\n')
            f_out.write(', '.join(map(str, training_set.test_set_ids)))
            f_out.write('\n## Features extracters used:')
            f_out.write( ''.join(['\n- ' + fe.__class__.__name__ for fe in training_set.features_extracters]) )
            # f_out.write('\n## Used features\' name:') # TODO
            # f_out.write(  ) # TODO save the used features' name
            f_out.write('\n## Labeler used:')
            f_out.write('\n' + training_set.labeler.__class__.__name__)
            f_out.write('\n## True labeler used:')
            f_out.write('\n' + training_set.true_labeler.__class__.__name__ if training_set.true_labeler is not None else '\nNone')
            # f_out.write('\n## Labels list:') # TODO
            # f_out.write(  ) # TODO save the labels list
            f_out.write('\n## Models trained and their optimal parameters:')
            f_out.write( ''.join([f'\n- {m}: {m.best_params}' for m in self.models]) )

        # zip those two files and the Config & used Models descriptions
        with zipfile.ZipFile(archive_filename, 'w') as f_out:

            # serialize each model
            for model in self.models:
                model_filename, model_tp_filename = model.save()

                # write serialized RecommendationModel instance to zip archive & clean up
                f_out.write(model_filename, os.path.split(model_filename)[-1], compress_type=zipfile.ZIP_DEFLATED)
                os.remove(model_filename)
                
                # write serialized trained_pipeline to zip archive & clean up
                f_out.write(model_tp_filename, os.path.split(model_tp_filename)[-1], compress_type=zipfile.ZIP_DEFLATED)
                os.remove(model_tp_filename)
                
            # save the TrainResults instance & info file to the archive
            f_out.write(pickle_filename, os.path.split(pickle_filename)[-1], compress_type=zipfile.ZIP_DEFLATED)
            f_out.write(info_filename, os.path.split(info_filename)[-1], compress_type=zipfile.ZIP_DEFLATED)
            
            # save the Config & used Models descriptions files to the archive
            for filepath in [*Utils.get_files_from_dir(RecommendationModel.MODELS_DESCRIPTION_DIR),
                             *Utils.get_files_from_dir(normp('Config/'))]:
                if not any(s in filepath for s in ['_template.py', '__pycache__']):
                    f_out.write(filepath, compress_type=zipfile.ZIP_DEFLATED)

        # clean up
        os.remove(pickle_filename)
        os.remove(info_filename)


    # static methods

    def load(id): # TODO fix: not working w/ new version
        """
        Loads a TrainResults instance from disk.

        Keyword arguments: 
        id -- id of the TrainResults instance to load

        Return: 
        TrainResults instance
        """
        archive_filename, pickle_filename, _ = TrainResults._get_filenames(id)
        with zipfile.ZipFile(archive_filename, 'r') as archive:
            with archive.open(os.path.split(pickle_filename)[-1], 'r') as pickle_file:
                return pickle.load(pickle_file)
    
    def _get_filenames(id):
        """
        Returns the filename of a TrainResults instance from its id.

        Keyword arguments: 
        id -- id of the TrainResults instance

        Return: 
        1. Filename of a TrainResults instance's save archive
        2. Filename of a TrainResults instance's pickle file
        3. Filename of a TrainResults instance's info file
        """
        return (
            normp(TrainResults.RESULTS_DIR + f'/{id}.zip'), # archive's filename
            normp(TrainResults.RESULTS_DIR + f'/{id}_serialized.p'), # pickle filename
            normp(TrainResults.RESULTS_DIR + f'/{id}_info.txt'), # info filename
        )