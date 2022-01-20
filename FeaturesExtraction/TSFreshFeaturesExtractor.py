"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
TSFreshFeaturesExtractor.py
@author: @chacungu
"""

import itertools
import numpy as np
import os
from os.path import normpath as normp
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from FeaturesExtraction.AbstractFeaturesExtractor import AbstractFeaturesExtractor
from Utils.Utils import Utils

class TSFreshFeaturesExtractor(AbstractFeaturesExtractor):
    """
    Singleton class which computes features from the TSFresh library.
    """

    FEATURES_FILENAMES_ID = '_tsfresh'


    # constructor

    def __new__(cls, *args, **kwargs):
        if 'caller' in kwargs and kwargs['caller'] == 'get_instance':
            return super(TSFreshFeaturesExtractor, cls).__new__(cls)
        raise Exception('Singleton class cannot be instantiated. Please use the static method "get_instance".')

    def __init__(self, *args, **kwargs):
        super().__init__()


    # public methods

    def extract(self, dataset):
        """
        Extracts and saves as CSV the features of the given data set's time series.
        
        Keyword arguments:
        dataset -- Dataset objects containing the time series from which features must be extracted
        
        Return:
        Updated Dataset object
        """
        timeseries = dataset.load_timeseries(transpose=False) # /!\ transpose False

        print('Running TSFresh on dataset %s.' % dataset.name)
        features_df = self.extract_from_timeseries(timeseries, dataset.nb_timeseries, dataset.timeseries_length)
        
        # save features as CSV
        dataset.save_features(self, features_df)
        return dataset

    def extract_from_timeseries(self, timeseries, nb_timeseries, timeseries_length):
        """
        Extracts the given time series' features.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series ( /!\ each column is a time series)
        nb_timeseries -- number of time series
        timeseries_length -- time series' length
        
        Return:
        Pandas DataFrame containing the time series' features ( /!\ each row is a time series' feature vector)
        """
        # prepare data to be used in tsfresh
        # DataFrame used as input of tsfresh:
        # time series id, time index, measured feature 1, measured feature 2, etc.
        # since in each data set contains time series of a single feature each (e.g. temperature in different cities),
        # we always have the three following columns: time series id, time index, values

        tsfresh_df = pd.DataFrame(columns=['Time Series ID', 'Time', 'Values'])
        timeseries_ids = [list(id for _ in range(timeseries_length)) 
                          for id in range(1, nb_timeseries+1)]
        tsfresh_df['Time Series ID'] = list(itertools.chain.from_iterable(timeseries_ids))
        times = [timeseries.index.tolist() for _ in range(nb_timeseries)]
        tsfresh_df['Time'] = list(itertools.chain.from_iterable(times))
        tsfresh_df['Values'] = np.array([timeseries[col].tolist() for col in timeseries.columns]).flatten()
        
        # extract features for the data set's time series
        features_df = extract_features(tsfresh_df, 
                                       column_id='Time Series ID', 
                                       column_sort='Time', n_jobs=os.cpu_count())
        
        # tsfresh imputation: remove NaNs, impute some missing values
        # features_df = impute(features_df) # this seems to hurt the performances
        
        features_df['Time Series ID'] = list(range(0, nb_timeseries))
        return features_df

    def save_features(self, dataset_name, features):
        """
        Saves the given features to CSV.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the features belong
        features -- Pandas DataFrame containing the features to save. Each row is a feature's vector.
                    Columns: Time Series ID, Feature 1's name, Feature 2's name, ...
        
        Return: -
        """
        features_filename = self._get_features_filename(dataset_name)
        features.to_csv(features_filename, index=False)
        
    def load_features(self, dataset):
        """
        Loads the features of the given data set's name.
        
        Keyword arguments: 
        dataset -- Dataset object to which the features belong
        
        Return: 
        Pandas DataFrame containing the data set's features. Each row is a time series feature vector.
        Columns: Time Series ID, Feature 1's name, Feature 2's name, ...
        """
        # load clusters features
        features_filename = self._get_features_filename(dataset.name)
        features_df = pd.read_csv(features_filename)

        # remove columns that only have 0s
        # features_df = features_df.T[features_df.any()].T # probably not needed (and possibly not a good idea)

        return features_df

    
    # private methods

    def _get_features_filename(self, dataset_name):
        """
        Returns the filename of the features for the given data set's name.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the features belong
        
        Return: 
        Filename of the features for the given data set's name.
        """
        return normp(
            AbstractFeaturesExtractor.FEATURES_DIR + \
            f'/{dataset_name}{TSFreshFeaturesExtractor.FEATURES_FILENAMES_ID}{AbstractFeaturesExtractor.FEATURES_APPENDIX}')

    
    # static methods

    @classmethod
    def get_instance(cls):
        """
        Returns the single instance of this class.
        
        Keyword arguments: - (no args required, cls is provided automatically since this is a classmethod)
        
        Return: 
        Single instance of this class.
        """
        return super().get_instance(cls)