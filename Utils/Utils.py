"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
Utils.py
@author: @chacungu
"""

from os.path import normpath as normp
import yaml

class Utils:
    """
    Static class with utilitary methods.
    """

    @staticmethod
    def read_conf_file(conf_name):
        """
        Loads a YAML configuration file.
        
        Keyword arguments:
        conf_name -- name of the configuration file (e.g. 'datasets', or 'clustering')
        
        Return:
        
        """
        path_prefix = './Config/'
        path_suffix = '_config.yaml'

        filename = normp(path_prefix + conf_name + path_suffix)
        try:
            with open(filename, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError('Configuration file %s not found.' % filename)