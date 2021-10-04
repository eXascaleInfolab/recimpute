"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
AbstractFeaturesExtracter.py
@author: @chacungu
"""

import abc

class AbstractFeaturesExtracter(metaclass=abc.ABCMeta):
    """
    Abstract features extracting class used to extract time series features and handle those features.
    """
    
    FEATURES_APPENDIX = '_features.csv'
    _INSTANCE = None


    # constructor

    def __init__(self):
        pass


    # public methods

    @abc.abstractmethod
    def extract(self, dataset):
        pass

    @abc.abstractmethod
    def save_features(self, dataset_name, features):
        pass

    @abc.abstractmethod
    def load_features(self, dataset):
        pass

    @abc.abstractmethod
    def are_features_created(self):
        pass


    # private methods

    #def


    # static methods

    @abc.abstractmethod
    def get_instance():
        pass