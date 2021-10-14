"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
AbstractLabeler.py
@author: @chacungu
"""

import abc
import os

from Utils.SingletonClass import SingletonClass

class AbstractLabeler(SingletonClass, metaclass=abc.ABCMeta):
    """
    Abstract Labeler class used to label time series and handle those labels.
    """
    
    LABELS_APPENDIX = '_labels.csv'
    

    @abc.abstractmethod
    def label_all_datasets(self, datasets):
        pass

    @abc.abstractmethod
    def label(self, dataset):
        pass

    @abc.abstractmethod
    def get_default_properties(self):
        pass

    @abc.abstractmethod
    def save_labels(self, dataset_name, labels):
        pass

    @abc.abstractmethod
    def load_labels(self, dataset, properties):
        pass

    @abc.abstractmethod
    def _get_labels_filename(self, dataset_name):
        pass

    def are_labels_created(self, dataset_name):
        """
        Checks whether the labels of the specified data set exist or not.
        
        Keyword arguments: 
        dataset_name -- name of the data set for which we check if the labels exist
        
        Return: 
        True if the labels have already been computed and saved as CSV, false otherwise.
        """
        labels_filename = self._get_labels_filename(dataset_name)
        return os.path.isfile(labels_filename)