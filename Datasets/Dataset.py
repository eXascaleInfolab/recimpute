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
import yaml
import zipfile

from Utils.Utils import Utils

class Dataset:
    """
    Class which handles a time series data set.
    """

    RW_DS_PATH = normp('./Datasets/RealWorld/')
    CONF = Utils.read_conf_file('datasets')

    def __init__(self, archive_name):
        self.rw_ds_filename = archive_name
        self.name = archive_name[:-4]

    def load_realworld(self, transpose=False):
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

    @staticmethod
    def instantiate_all_realworld(transpose=False):
        """
        Instantiates real-world data set from the Datasets/RealWorld folder.
        Uses the Datasets conf file to define which data set use.
        
        Keyword arguments:
        transpose -- transpose the data set if true (default False)
        
        Return:
        List of Dataset objects
        """
        rw_ds = [Dataset(ds_filename) 
                 for ds_filename in os.listdir(Dataset.RW_DS_PATH)
                 if Dataset.CONF['USE_ALL'] or ds_filename in Dataset.CONF['USE_LIST']]
        
        # check: either use all data sets listed in the folder
        # ... or verify that all data sets listed in the conf file have been found and loaded
        assert Dataset.CONF['USE_ALL'] or len(rw_ds) == len(Dataset.CONF['USE_LIST'])

        return rw_ds
