"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
Utils.py
@author: @chacungu
"""

from collections import ChainMap
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
        Dictionary containing all parameters.
        """
        path_prefix = './Config/'
        path_suffix = '_config.yaml'

        filename = normp(path_prefix + conf_name + path_suffix)
        try:
            with open(filename, 'r') as f:
                conf = yaml.safe_load(f)

                # merge params' values loaded as list of dicts into a single dict
                is_param_nested_dict = lambda param_val: isinstance(param_val, list) and all(isinstance(sub_val, dict) for sub_val in param_val)
                for param, value in conf.items():
                    if is_param_nested_dict(value):
                        conf[param] = dict(ChainMap(*value))
                        
                return conf
        except FileNotFoundError:
            raise FileNotFoundError('Configuration file %s not found.' % filename)