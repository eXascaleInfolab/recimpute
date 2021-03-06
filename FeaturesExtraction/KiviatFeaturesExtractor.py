"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
KiviatFeaturesExtractor.py
@author: @chacungu
"""

from adtk.aggregator import OrAggregator
from adtk.data import validate_series
from adtk.detector import QuantileAD, InterQuartileRangeAD, GeneralizedESDTestAD, PersistAD
import numpy as np
import os
from os.path import normpath as normp
import pandas as pd
import sys

from FeaturesExtraction.AbstractFeaturesExtractor import AbstractFeaturesExtractor
from Utils.Utils import Utils

class KiviatFeaturesExtractor(AbstractFeaturesExtractor):
    """
    Singleton class which computes few "simple" features: Time series' length, an irregularity score, and pairwise correlation.
    """

    FEATURES_FILENAMES_ID = '_kiviat'


    # constructor

    def __new__(cls, *args, **kwargs):
        if 'caller' in kwargs and kwargs['caller'] == 'get_instance':
            return super(KiviatFeaturesExtractor, cls).__new__(cls)
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
        timeseries = dataset.load_timeseries(transpose=True)

        features = []
        for cluster, cluster_id, _ in dataset.yield_all_clusters(timeseries): # for each cluster in the data set
            # compute the cluster's time series features
            # feature 1: length
            length = dataset.timeseries_length

            # feature 2: irregularity score
            irregularity = self._get_irregularity_score(cluster)

            # feature 3: pairwise correlation
            corr_matrix = np.array(cluster.T.corr())
            corr_upper_values = np.array(Utils.strictly_upper_triang_val(corr_matrix))
            correlation = corr_upper_values[~np.isnan(corr_upper_values)] # (remove NaNs)

            features.append((cluster_id, length, irregularity, correlation))

        features_df = pd.DataFrame(features, columns=['Cluster ID', 'Length', 'Irregularity', 'Correlation'])
        
        # save features as CSV
        dataset.save_features(self, features_df)
        return dataset

    def extract_from_timeseries(self, timeseries):
        raise Exception('The KiviatFeaturesExtractor is not capable of extracting features on time series that are not clustered.'
                      + ' Please use the "extract" method to extract features for a whole clustered data set.')

    def save_features(self, dataset_name, features):
        """
        Saves the given features to CSV.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the features belong
        features -- Pandas DataFrame containing the features to save. Four columns: Cluster ID, Length, Irregularity, Correlation.
        
        Return: -
        """
        features_filename = self._get_features_filename(dataset_name)
        np.set_printoptions(threshold=sys.maxsize)
        features.to_csv(features_filename, index=False)
        np.set_printoptions(threshold=1000)
        
    def load_raw_features(self, dataset):
        """
        Loads the raw features of the given data set's name.
        
        Keyword arguments: 
        dataset -- Dataset object to which the features belong
        
        Return: 
        Pandas DataFrame containing the data set's raw features. Each row is a time series feature vector.
        Columns: Time Series ID, Cluster ID, Length, Irregularity, Correlation
        """
        # load clusters features
        features_filename = self._get_features_filename(dataset.name)
        all_clusters_features = pd.read_csv(features_filename, index_col='Cluster ID', 
                                            converters={'Correlation': lambda instr: np.fromstring(instr[1:-1], sep=' ')})

        
        # propagate cluster features to time series
        timeseries_features = []
        for _, row in dataset.load_cassignment(dataset.clusterer).iterrows():
            tid = row['Time Series ID']
            cid = row['Cluster ID']
            cluster_features = all_clusters_features.loc[[cid]]
            timeseries_features.append((tid, cid, *cluster_features.values[0]))

        timeseries_features_df = pd.DataFrame(timeseries_features, 
                                              columns=['Time Series ID', 'Cluster ID', *cluster_features.columns.values.tolist()])
        return timeseries_features_df

    def load_features(self, dataset):
        """
        Loads the features of the given data set's name.
        
        Keyword arguments: 
        dataset -- Dataset object to which the features belong
        
        Return: 
        Pandas DataFrame containing the data set's features. Each row is a time series feature vector.
        Columns: Time Series ID, Cluster ID, Feature 1's name, Feature 2's name, ...
        """
        features = self.load_raw_features(dataset)
        features['Correlation'] = features['Correlation'].apply(lambda array: np.median(array))

        features.columns = map(
            lambda col_name: col_name + KiviatFeaturesExtractor.FEATURES_FILENAMES_ID if col_name not in ['Time Series ID', 'Cluster ID'] else col_name, 
            features.columns
        )
        return features
    
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
            f'/{dataset_name}{KiviatFeaturesExtractor.FEATURES_FILENAMES_ID}{AbstractFeaturesExtractor.FEATURES_APPENDIX}')

    def _get_irregularity_score(self, timeseries):
        """
        Computes and returns the number of anomalies found over all time series of a cluster divided by the length 
        of those time series.
        
        Keyword arguments:
        timeseries -- DataFrame containing the time series (each row is a time series)
        
        Return:
        Float: number of anomalies found over all time series of a cluster divided by the length of those time series.
        """
        all_anomalies = []
        for _, s_ in timeseries.iterrows(): # for each time series
            s = validate_series(s_)

            detectors = [
                QuantileAD(high=0.99, low=0.01),
                #    compares each time series value with historical quantiles
                
                InterQuartileRangeAD(c=1.5),
                #    based on simple historical statistics, based on interquartile range (IQR)
                
                GeneralizedESDTestAD(alpha=0.3),
                #    detects anomaly based on generalized extreme Studentized deviate (ESD) test
                
                PersistAD(c=3.0, side='positive'),
                #    compares each time series value with its previous values
            ]

            try:
                all_anomalies.extend([d.fit_detect(s) for d in detectors])
            except:
                all_anomalies.extend([np.zeros(len(s), dtype=bool) for d in detectors])
            
        all_anomalies = OrAggregator().aggregate(pd.DataFrame(np.array(all_anomalies).T, index=s.index))
        all_anomalies_bool = all_anomalies.astype('bool')
        anomalies_distribution = all_anomalies_bool.value_counts(normalize=False)
        anomalies_distribution = anomalies_distribution[True] if True in anomalies_distribution else 0
        anomalies_percentage = anomalies_distribution / len(s)
        
        return anomalies_percentage


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