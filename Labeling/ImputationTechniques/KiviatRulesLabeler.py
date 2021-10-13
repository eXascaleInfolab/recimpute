"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
KiviatRulesLabeler.py
@author: @chacungu
"""

import pandas as pd
import numpy as np
from os.path import normpath as normp

from Labeling.AbstractLabeler import AbstractLabeler
from FeaturesExtraction.KiviatFeaturesExtracter import KiviatFeaturesExtracter
from Utils.Utils import Utils

class KiviatRulesLabeler(AbstractLabeler):
    """
    Singleton labeler class which extracts and uses rules from Kiviat diagrams about imputation techniques. Provides methods 
    to label time series and handle those labels.

    The Kiviat diagrams are presented in "Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques
    in Time Series" by Khayati et al. in 2020.
    """

    LABELS_DIR = normp('./Labeling/ImputationTechniques/labels/')
    LABELS_FILENAMES_ID = '_krl'
    KIVIAT_VALUES = {
        'cdrec': 
            {'efficient': 1, 'large_ts': 3, 'irregular_ts': 3, 'mixed_corr': 5, 'high_corr': 3},
        'dynammo': 
            {'efficient': 0, 'large_ts': 0, 'irregular_ts': 5, 'mixed_corr': 4, 'high_corr': 3},
        'softimp': 
            {'efficient': 1, 'large_ts': 3, 'irregular_ts': 4, 'mixed_corr': 3, 'high_corr': 2},
        'svdimp': 
            {'efficient': 1, 'large_ts': 4, 'irregular_ts': 3, 'mixed_corr': 4, 'high_corr': 3},
        'stmvl':
            {'efficient': 0, 'large_ts': 2, 'irregular_ts': 2, 'mixed_corr': 1, 'high_corr': 5},
        'trmf': 
            {'efficient': 0, 'large_ts': 1, 'irregular_ts': 3, 'mixed_corr': 4, 'high_corr': 3},
    }
    CONF = Utils.read_conf_file('kiviatruleslabeler')

    # create necessary directories if not there yet
    Utils.create_dirs_if_not_exist([LABELS_DIR])


    # constructor

    def __new__(cls, *args, **kwargs):
        if 'caller' in kwargs and kwargs['caller'] == 'get_instance':
            return super(KiviatRulesLabeler, cls).__new__(cls)
        raise Exception('Singleton class cannot be instantiated. Please use the static method "get_instance".')

    def __init__(self, *args, **kwargs):
        super().__init__()


    # public methods

    def label_all_datasets(datasets):
        """
        Labels each cluster from the given list of data sets using the Kiviat rules.
        
        Keyword arguments:
        datasets -- List of Dataset objects containing time series to label
        
        Return:
        List of updated Dataset objects
        """
        updated_datasets = []
        
        # for each data set
        for dataset in datasets:
            dataset = self.label(dataset)
            updated_datasets.append(dataset)
        return updated_datasets

    def label(self, dataset):
        """
        Labels each cluster from the given data set using the Kiviat rules.
        
        Keyword arguments:
        dataset -- Dataset object containing time series to label
        
        Return:
        Updated Dataset object
        """
        # load the cluster's features needed to apply the Kiviat rules
        features_extracter = KiviatFeaturesExtracter.get_instance()
        if not features_extracter.are_features_created(dataset.name):
            dataset = features_extracter.extract(dataset)
        features = features_extracter.load_raw_features(dataset)
        # sample 1 features row per Cluster ID 
        # (all time series of a same cluster have the same features w/ KiviatFeaturesExtracter)
        features = features.groupby('Cluster ID', group_keys=False).apply(lambda df: df.sample(1))

        # features: DataFrame - Cluster ID, Length, Irregularity, Correlation

        features['STD Correlation'] = features['Correlation'].apply(np.std)
        features['Median Correlation'] = features['Correlation'].apply(np.median)
        features = features.drop(columns=['Correlation'])
        
        # features: DataFrame - Cluster ID, Length, Irregularity, STD Correlation, Median Correlation
        
        # save labels
        dataset.save_labels(self, features)
        return dataset

    def get_labels_possible_properties(self):
        """
        Returns a dict containing the possible properties labels generated with this labeler can have.
        
        Keyword arguments: -
        
        Return: 
        Dict containing the possible properties labels generated with this labeler can have.
        """
        return KiviatRulesLabeler.CONF['POSSIBLE_LBL_PROPERTIES']

    def save_labels(self, dataset_name, labels):
        """
        Saves the given labels to CSV.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the labels belong
        labels -- Pandas DataFrame containing the labels to save. Two columns: Cluster ID, Benchmark Results.
        
        Return: -
        """
        labels_filename = self._get_labels_filename(dataset_name)
        labels.to_csv(labels_filename, index=False)

    def load_labels(self, dataset, properties):
        """
        Loads the labels of the given data set's name and defined by the specified properties.
        
        Keyword arguments: 
        dataset -- Dataset object to which the labels belong
        properties -- dict specifying the labels' properties
        
        Return: 
        1. Pandas DataFrame containing the data set's labels. Two columns: Time Series ID and Label.
        2. List of all possible labels value
        """
        # load clusters labels
        labels_filename = self._get_labels_filename(dataset.name)
        all_clusters_features = pd.read_csv(labels_filename, index_col='Cluster ID')
        
        # create labels from kiviat features
        timeseries_labels = []
        for _, row in dataset.load_cassignment().iterrows():
            tid = row['Time Series ID']
            cid = row['Cluster ID']
            cluster_features = all_clusters_features.loc[[cid]]
            label = self._apply_rules(cluster_features)
            timeseries_labels.append((tid, label))

        timeseries_labels_df = pd.DataFrame(timeseries_labels, columns=['Time Series ID', 'Label'])
        return timeseries_labels_df, KiviatRulesLabeler.CONF['ALGORITHMS_LIST']


    # private methods

    def _get_labels_filename(self, dataset_name):
        """
        Returns the filename of the labels for the given data set's name.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the labels belong
        
        Return: 
        Filename of the labels for the given data set's name.
        """
        return normp(
            KiviatRulesLabeler.LABELS_DIR + \
            f'/{dataset_name}{KiviatRulesLabeler.LABELS_FILENAMES_ID}{AbstractLabeler.LABELS_APPENDIX}')

    def _apply_rules(self, cluster_features):
        """
        Applies the Kiviat rules to define the cluster's time series label.
        
        Keyword arguments: 
        cluster_features -- DataFrame with a single row which contains a cluster's Kiviat features. 
                            Columns: Cluster ID, Length, Irregularity, STD Correlation, Median Correlation
        
        Return: 
        Label of a cluster's time series
        """
        # apply thresholds
        thresholds = KiviatRulesLabeler.CONF['FEATURES_THRESHOLDS']
        binary_cluster_features = {
            'large_ts': int(cluster_features['Length'] >= thresholds['large_ts']),
            'irregular_ts': int(cluster_features['Irregularity'] > thresholds['irregular_ts']),
            'mixed_corr': int(cluster_features['STD Correlation'] > thresholds['mixed_corr']),
            'high_corr': int(cluster_features['Median Correlation'] > thresholds['high_corr']),
        }

        features_weights = KiviatRulesLabeler.CONF['FEATURES_WEIGHTS']
        kiviat_values = KiviatRulesLabeler.KIVIAT_VALUES

        # compute a score for each algorithm
        scores = {}
        for algo in KiviatRulesLabeler.CONF['ALGORITHMS_LIST']:
            score = features_weights['efficient'] * kiviat_values[algo]['efficient']
            for name, value in binary_cluster_features.items():
                score += value * features_weights[name] * kiviat_values[algo][name]
            scores[algo] = score
        
        # label is the algorithms producing the highest score
        return max(scores.keys(), key=(lambda key: scores[key])) # return algorithm with highest score


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