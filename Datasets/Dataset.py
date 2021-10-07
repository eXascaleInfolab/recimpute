"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
Dataset.py
@author: @chacungu
"""

import os
from os.path import normpath as normp
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import re
from tqdm import tqdm
import yaml
import zipfile

from Utils.Utils import Utils

class Dataset:
    """
    Class which handles a time series data set.
    """

    RW_DS_PATH = normp('./Datasets/RealWorld/')
    CASSIGNMENTS_DIR = normp('./Clustering/Results/cassignments/')
    CASSIGNMENTS_APPENDIX = '_cassignments.csv'
    CONF = Utils.read_conf_file('datasets')

    # create necessary directories if not there yet
    Utils.create_dirs_if_not_exist([CASSIGNMENTS_DIR])


    # constructor

    def __init__(self, archive_name):
        self.rw_ds_filename = archive_name
        self.name = archive_name[:-4]
        self.nb_timeseries, self.timeseries_length = self.load_timeseries(transpose=True).shape

        self.cids = None
        if self.are_clusters_created():
            cassignment = self.load_cassignment()
            self.cids = cassignment['Cluster ID'].unique().tolist()


    # public methods

    def get_cluster_by_id(self, timeseries, cluster_id, cassignment=None):
        """
        Returns the time series belonging to the specified cluster's id.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series (each row is a time series).
        cluster_id -- cluster id (int) of which the time series that must be returned belong to.
        cassignment -- Pandas DataFrame containing clusters' assignment of the data set's time series. 
                       Its index is the same as the real world data set of this object. The associated 
                       values are the clusters' id to which are assigned the time series (default None, if None, loads it).
        
        Return:
        Pandas DataFrame containing the time series belonging to the specified cluster's id (each row is a time series).
        """
        if cassignment is None:
            cassignment = self.load_cassignment()
        return timeseries.loc[cassignment['Time Series ID'][cassignment['Cluster ID'] == cluster_id]]

    def yield_all_clusters(self, timeseries, cassignment=None):
        """
        Yields the time series of each cluster in a Pandas DataFrame.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series (each row is a time series).
        cluster_id -- cluster id (int) of which the time series that must be returned belong to.
        cassignment -- Pandas DataFrame containing clusters' assignment of the data set's time series. 
                       Its index is the same as the real world data set of this object. The associated 
                       values are the clusters' id to which are assigned the time series (default None, if None, loads it).
        
        Return:
        1. Pandas DataFrame containing the time series belonging to the specified cluster's id (each row is a time series).
        2. Cluster id (int) of the cluster being returned.
        3. Pandas DataFrame containing clusters' assignment of the data set's time series. Its index is the same as the real 
           world data set of this object. The associated values are the clusters' id to which are assigned the time series.
        """
        # load clusters assignment
        if cassignment is None:
            cassignment = self.load_cassignment()
        for cluster_id in cassignment['Cluster ID'].unique(): # for each cluster ID present in this dataset
            # retrieve time series assigned to this cluster
            cluster = self.get_cluster_by_id(timeseries, cluster_id, cassignment)
            yield cluster, cluster_id, cassignment

    def load_timeseries(self, transpose=False):
        """
        Loads time series which are stored in a .txt file with either a .info or a .index file describing the dates used as index.
        
        Keyword arguments:
        transpose -- transpose the data set if true (default False)
        
        Return:
        Pandas DataFrame containing the time series
        """
        ds_filename_ext = ('txt', 'csv')
        index_filename_ext = 'index'
        info_filename_ext = 'info' 

        # load archive
        with zipfile.ZipFile(normp(Dataset.RW_DS_PATH + '/' + self.rw_ds_filename), 'r') as archive:
            prefix = f'{self.name}/' if archive.namelist()[0] == self.name + '/' else ''

            r = re.compile(f'.*{self.name}.*\.[{ds_filename_ext[0]}|\
                                               {ds_filename_ext[1]}|\
                                               {index_filename_ext}|\
                                               {info_filename_ext}]', re.IGNORECASE)
            filenames = list(filter(r.match, archive.namelist()))
            _get_filename = lambda l, ext: next(x for x in l if x.endswith(ext))

            # load data set
            ds_filename = _get_filename(filenames, ds_filename_ext)
            dataset = pd.read_csv(archive.open(ds_filename), sep=' ', header=None)
            
            if self._is_datetime_col(dataset[0]): # if first column is of type datetime: use it as index
                dataset['DateTime'] = pd.to_datetime(dataset[0])
                dataset = dataset.drop(columns=[0])
                dataset = dataset.set_index('DateTime')
                dataset.columns = pd.RangeIndex(dataset.columns.size)
            else: # else try to load an index or info file containing information about the data set's index
                # load data set's index (date range)
                try:
                    # index: each point's date is specified in file
                    # index file: 1 column DataFrame (1 x nb_points_in_time_series), each row is a Date
                    index_filename = _get_filename(filenames, index_filename_ext)
                    index = pd.read_csv(archive.open(index_filename), sep=' ', header=None, parse_dates=True)
                    dataset['DateTime'] = index[0].tolist()
                    dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
                    dataset = dataset.set_index('DateTime')
                except StopIteration: # index file does not exist
                    # info: only start date, periods and freq are given, date range is created from this
                    # info file: 2 rows: header (start, periods, freq) and content (Date, int, Char)
                    try:
                        info_filename = _get_filename(filenames, info_filename_ext)
                        date_range = pd.read_csv(archive.open(info_filename), sep=' ', parse_dates=True)
                        date_range = dict(zip(date_range.columns, date_range.values[0]))
                    except StopIteration:
                        date_range = {'start': '1900-1-1', 'periods': len(dataset.index), 'freq': '1s'}
                    dataset = dataset.set_index(pd.date_range(**date_range))

        return dataset if not transpose else dataset.T

    def save_cassignment(self, cassignment):
        """
        Loads the Pandas Dataframe containing the clusters' assignment of this data set. 
        
        Keyword arguments:
        cassignment -- Pandas DataFrame containing clusters' assignment of the data set's time series. 
                       Its index is the same as the real world data set of this object. The associated 
                       values are the clusters' id to which are assigned the time series.
        
        Return: -
        """
        cassignment_filename = self._get_cassignment_filename()
        self.cids = cassignment['Cluster ID'].unique().tolist()
        cassignment.to_csv(cassignment_filename, index=False)

    def load_cassignment(self):
        """
        Loads the Pandas Dataframe containing the clusters' assignment of this data set. 
        
        Keyword arguments: -
        
        Return:
        Pandas DataFrame containing clusters' assignment of the data set's time series. Its index is the same 
        as the real world data set of this object. The associated values are the clusters' id to which are
        assigned the time series.
        """
        cassignment_filename = self._get_cassignment_filename()
        clusters_assignment = pd.read_csv(cassignment_filename)
        return clusters_assignment

    def are_clusters_created(self):
        """
        Checks whether the clusters exist or not.
        
        Keyword arguments: -
        
        Return: 
        True if the clusters have already been created and saved as CSV, false otherwise.
        """
        cassignment_filename = self._get_cassignment_filename()
        return os.path.isfile(cassignment_filename)

    def get_space_complexity(self):
        """
        Computes and returns the space complexity of the data set's time series.
        
        Keyword arguments: -
        
        Return:
        Space complexity of the data set's time series
        """
        return self.nb_timeseries * self.timeseries_length

    def load_labels(self, labeler, properties):
        """
        Loads the labels created using the given labeler and defined by the specified properties.
        
        Keyword arguments: 
        labeler -- labeler instance used to create the labels to load
        properties -- dict specifying the labels' properties
        
        Return: 
        1. Pandas DataFrame containing the data set's labels. Two columns: Time Series ID and Label.
        2. List of all possible labels value
        """
        return labeler.load_labels(self, properties)

    def save_labels(self, labeler, labels):
        """
        Saves the given labels to CSV.
        
        Keyword arguments: 
        labeler -- labeler instance used to create the labels to save
        labels -- Pandas DataFrame containing the labels to save. 
        
        Return: -
        """
        labeler.save_labels(self.name, labels)

    def load_features(self, features_extracter):
        """
        Loads the features created using the given extracter.
        
        Keyword arguments: 
        features_extracter -- features extracter instance used to create the features to load
        
        Return: 
        Pandas DataFrame containing the data set's features. Each row is a time series feature vector.
        Columns: Time Series ID, (Cluster ID), Feature 1's name, Feature 2's name, ...
        """
        return features_extracter.load_features(self)

    def save_features(self, features_extracter, features):
        """
        Saves the given features to CSV.
        
        Keyword arguments: 
        features_extracter -- features extracter instance used to create the features to save
        features -- Pandas DataFrame containing the features to save. 
        
        Return: -
        """
        features_extracter.save_features(self.name, features)


    # private methods

    def __repr__(self):
        return self.name

    def _get_cassignment_filename(self):
        """
        Returns the cassignment filename of this data set.
        
        Keyword arguments: -
        
        Return:
        cassignment filename
        """
        return normp(Dataset.CASSIGNMENTS_DIR + f'/{self.name}{Dataset.CASSIGNMENTS_APPENDIX}')
    
    def _is_datetime_col(self, col):
        """
        Checks if a Pandas Series is of type date time.
        
        Keyword arguments:
        col -- Pandas series
        
        Return:
        True if the Pandas Series contains only date time objects False otherwise
        """
        if col.dtype == 'object':
            try:
                col = pd.to_datetime(col)
                return True
            except ValueError:
                return False
        return is_datetime(col)


    # static methods

    @staticmethod
    def instantiate_from_dir():
        """
        Instantiates multiple data set objects from the Datasets/RealWorld folder.
        Uses the Datasets conf file to define which data set use.
        
        Keyword arguments: -
        
        Return:
        List of Dataset objects
        """
        timeseries = [Dataset(ds_filename) 
                      for ds_filename in os.listdir(Dataset.RW_DS_PATH)
                      if Dataset.CONF['USE_ALL'] or ds_filename in Dataset.CONF['USE_LIST']]
        
        # check: either use all data sets listed in the folder
        # ... or verify that all data sets listed in the conf file have been found and loaded
        assert Dataset.CONF['USE_ALL'] or len(timeseries) == len(Dataset.CONF['USE_LIST'])

        return timeseries

    @staticmethod
    def yield_each_datasets_cluster(datasets):
        """
        One-by-one, yields each cluster of all given datasets.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series and their clusters assignment.
        
        Return: 
        1. Dataset object which contains the yielded cluster
        2. Pandas DataFrame containing all the time series of the data set (each row is a time series)
        3. Pandas DataFrame containing the time series belonging to one of the clusters to yield (each row is a time series).
        4. ID of the yielded cluster
        """
        for dataset in tqdm(datasets): # for each data set
            timeseries = dataset.load_timeseries(transpose=True) # load data set's time series
            clusters_assignment = dataset.load_cassignment() # load clusters assignment
            for cluster_id in clusters_assignment['Cluster ID'].unique(): # for each cluster
                cluster = dataset.get_cluster_by_id(timeseries, cluster_id, clusters_assignment)
                yield dataset, timeseries, cluster, cluster_id

    @staticmethod
    def yield_each_datasets_cluster_id(datasets):
        """
        One-by-one, yields each cluster id of all given datasets.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series and their clusters assignment.
        
        Return: 
        ID of the yielded cluster
        """
        for dataset in datasets: # for each data set
            for cid in dataset.cids:
                yield cid