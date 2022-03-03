"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
Catch22FeaturesExtractor.py
@author: @chacungu
"""

from catch22 import catch22_all
from os.path import normpath as normp
import pandas as pd

from FeaturesExtraction.AbstractFeaturesExtractor import AbstractFeaturesExtractor

class Catch22FeaturesExtractor(AbstractFeaturesExtractor):
    """
    Singleton class which computes features from the Catch22 library.
    """

    FEATURES_FILENAMES_ID = '_catch22'


    # constructor

    def __new__(cls, *args, **kwargs):
        if 'caller' in kwargs and kwargs['caller'] == 'get_instance':
            return super(Catch22FeaturesExtractor, cls).__new__(cls)
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

        print('Running Catch22 on dataset %s.' % dataset.name)
        features_df = self.extract_from_timeseries(timeseries)
        
        # save features as CSV
        dataset.save_features(self, features_df)
        return dataset

    def extract_from_timeseries(self, timeseries):
        """
        Extracts the given time series' features.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series ( /!\ each column is a time series)
        
        Return:
        Pandas DataFrame containing the time series' features ( /!\ each row is a time series' feature vector)
        """
        map_catch22_res_to_dict = lambda res: {name: val for name, val in zip(res['names'], res['values'])}

        # extract features for the data set's time series
        features_df = pd.DataFrame(
            [map_catch22_res_to_dict(catch22_all(ts)) for ts in timeseries.T.to_numpy()]
        )
        features_df['Time Series ID'] = list(range(0, timeseries.shape[1]))
        
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
            f'/{dataset_name}{Catch22FeaturesExtractor.FEATURES_FILENAMES_ID}{AbstractFeaturesExtractor.FEATURES_APPENDIX}')

    
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