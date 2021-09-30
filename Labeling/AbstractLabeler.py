"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
AbstractLabeler.py
@author: @chacungu
"""

import abc

class AbstractLabeler(metaclass=abc.ABCMeta):
    """
    Abstract Labeler class used to label time series and handle those labels.
    """

    # public methods

    @abc.abstractmethod
    def label(self, datasets):
        pass


    # static methods

    @abc.abstractmethod
    def get_labels_possible_properties():
        pass

    @abc.abstractmethod
    def save_labels(dataset_name, labels):
        pass

    @abc.abstractmethod
    def load_labels(dataset_name, properties):
        pass