"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
KatsFeaturesExtractor.py
@author: @chacungu
"""

# from kats.consts import TimeSeriesData
# from kats.tsfeatures.tsfeatures import TsFeatures
import math
from os.path import normpath as normp
import pandas as pd
import warnings

from FeaturesExtraction.AbstractFeaturesExtractor import AbstractFeaturesExtractor

class KatsFeaturesExtractor(AbstractFeaturesExtractor):
    """
    Singleton class which computes features from the Kats library.
    """

    FEATURES_FILENAMES_ID = '_kats'


    # constructor

    def __new__(cls, *args, **kwargs):
        if 'caller' in kwargs and kwargs['caller'] == 'get_instance':
            return super(KatsFeaturesExtractor, cls).__new__(cls)
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

        print('Running Kats on dataset %s.' % dataset.name)
        try:
            features_df = self.extract_from_timeseries(timeseries)
        
            # save features as CSV
            dataset.save_features(self, features_df)
        except Exception as e:
            print('Got exception for dataset %s.' % dataset.name)
            print(e)

        return dataset

    def extract_from_timeseries(self, timeseries):
        """
        Extracts the given time series' features.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series ( /!\ each column is a time series)
        
        Return:
        Pandas DataFrame containing the time series' features ( /!\ each row is a time series' feature vector)
        """
        window_size = 20 if timeseries.shape[0] > 20 else math.floor(timeseries.shape[0] / 2)
        stl_period = acfpacf_lag = (timeseries.shape[0] // 2) - 2 if 7 >= timeseries.shape[0] // 2 else 7
        model = TsFeatures(window_size=window_size, stl_period=stl_period, acfpacf_lag=acfpacf_lag)
        features = []
        
        # extract features for the data set's time series
        for col in timeseries.columns:
            # prepare data to be used in Kats: 1 dataframe per time series: 2 cols - "time" and the sequence's values
            ts = timeseries[col]
            ts_df = pd.DataFrame(ts)
            ts_df['time'] = ts_df.index
            ts_df.index = range(ts_df.shape[0])
            
            ts_data = TimeSeriesData(ts_df)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features.append(model.transform(ts_data))

        features_df = pd.DataFrame(features)
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

        features_df.columns = map(
            lambda col_name: col_name + KatsFeaturesExtractor.FEATURES_FILENAMES_ID if col_name not in ['Time Series ID'] else col_name, 
            features_df.columns
        )
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
            f'/{dataset_name}{KatsFeaturesExtractor.FEATURES_FILENAMES_ID}{AbstractFeaturesExtractor.FEATURES_APPENDIX}')

    
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