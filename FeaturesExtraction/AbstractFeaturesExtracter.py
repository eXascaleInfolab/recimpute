"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
AbstractFeaturesExtracter.py
@author: @chacungu
"""

import abc
import os
from os.path import normpath as normp

from Utils.Utils import Utils
from Utils.SingletonClass import SingletonClass

class AbstractFeaturesExtracter(SingletonClass, metaclass=abc.ABCMeta):
    """
    Abstract features extracting class used to extract time series features and handle those features.
    """
    
    FEATURES_DIR = normp('./FeaturesExtraction/features/')
    FEATURES_APPENDIX = '_features.csv'

    # create necessary directories if not there yet
    Utils.create_dirs_if_not_exist([FEATURES_DIR])


    @abc.abstractmethod
    def extract(self, dataset):
        pass

    @abc.abstractmethod
    def extract_from_timeseries(self, dataset):
        pass

    @abc.abstractmethod
    def save_features(self, dataset_name, features):
        pass

    @abc.abstractmethod
    def load_features(self, dataset):
        pass

    @abc.abstractmethod
    def _get_features_filename(self, dataset_name):
        pass

    def are_features_created(self, dataset_name):
        """
        Checks whether the features of the specified data set exist or not.
        
        Keyword arguments: 
        dataset_name -- name of the data set for which we check if the features exist
        
        Return: 
        True if the features have already been computed and saved as CSV, false otherwise.
        """
        features_filename = self._get_features_filename(dataset_name)
        return os.path.isfile(features_filename)