"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
Utils.py
@author: @chacungu
"""

from collections import ChainMap, Counter
from glob import glob
import math
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import normpath as normp
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.utils.multiclass import unique_labels
from time import perf_counter
import warnings
import yaml

class Utils:
    """
    Static class with utilitary methods.
    """

    def custom_train_test_split(X, y=None, test_size=None, train_size=None, shuffle=True, stratify=None):
        """
        Split arrays or pandas DataFrame into random train and test subsets.
        
        TODO: deal with the extreme scenario where the number of single member classes is high enough to
              unbalance the test_size
        Keyword arguments:
        X -- array of pandas DataFrame to split
        y -- array of pandas DataFrame to split
        test_size -- proportion or number of data to include in the test
        train_size -- proportion or number of data to include in the train
        shuffle -- True if data should be shuffled, False otherwise
        stratify -- array-like, used as class labels if provided
        
        Return:
        Splitted arrays (train and test)
        """
        if not y is None or not stratify is None:
            labels = stratify if not stratify is None else y
            class_counts = Counter(labels)
            single_class_indices = [i for i, label in enumerate(labels) if class_counts[label] == 1]
            if len(single_class_indices) > 0:
                print('Warning: %i classe(s) contain a single member (%s). Consider adding more members or ignore this class to avoid further issues.\n' % (len(single_class_indices), pd.Series(labels).value_counts()[lambda x: x == 1].index.tolist()))
            multi_class_indices = [i for i, label in enumerate(labels) if class_counts[label] > 1]

            # identify classes with one member
            index_arr = lambda arr, indices: arr.iloc[indices] if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series) else arr[indices]
            X_train = index_arr(X, single_class_indices)
            if not y is None:
                y_train = index_arr(y, single_class_indices)
            
            # split the rest using train_test_split
            X_rest = index_arr(X, multi_class_indices)
            if not y is None:
                y_rest = index_arr(y, multi_class_indices)

            if not stratify is None:
                stratify_rest = index_arr(stratify, multi_class_indices)
            else:
                stratify_rest = None

            arrays_rest = [X_rest, y_rest] if not y is None else [X_rest]
            arrays_splitted = sklearn_train_test_split(
                *arrays_rest, test_size=test_size, train_size=train_size, stratify=stratify_rest, shuffle=shuffle,
            )
            if not y is None:
                X_train_rest, X_test, y_train_rest, y_test = arrays_splitted
            else:
                X_train_rest, X_test = arrays_splitted

            X_train = pd.concat((X_train, X_train_rest)) if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) else np.concatenate((X_train, X_train_rest), axis=0)
            if not y is None:
                y_train = pd.concat((y_train, y_train_rest)) if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else np.concatenate((y_train, y_train_rest), axis=0)
                return X_train, X_test, y_train, y_test
            else:
                return X_train, X_test
        else:
            return sklearn_train_test_split(X, test_size=test_size, shuffle=shuffle)

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
                def list_to_dict(parent_dict):
                    for param, value in parent_dict.items():
                        if is_param_nested_dict(value):
                            parent_dict[param] = dict(ChainMap(*value))
                            list_to_dict(parent_dict[param])
                        elif isinstance(value, dict):
                            list_to_dict(value)
                list_to_dict(conf)
                        
                return conf
        except FileNotFoundError:
            raise FileNotFoundError('Configuration file %s not found.' % filename)

    def create_dirs_if_not_exist(paths):
        """
        For each path in the given list, creates the directory at specified path if it does not exist yet.
        
        Keyword arguments:
        paths -- list of paths to the directories to create
        
        Return: -
        """
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def strictly_upper_triang_val(m):
        """
        Returns only the strictly upper triangular values from the given matrix.
        
        Keyword arguments:
        m -- 2d list (matrix) from which return the strictly upper triangular values
        
        Return: 
        List of strictly upper triangular values from the given matrix
        """
        return [m[i,j] for i in range(len(m)) for j in range(len(m[i])) if i < j]

    def get_files_from_dir(dir_path):
        """
        Returns a list of files' path from the specified directory.
        
        Keyword arguments:
        dir_path -- path to the directory to search files in
        
        Return: 
        List of files' path from the specified directory
        """
        glob_pattern = os.path.join(dir_path, '*')
        files = sorted(glob(glob_pattern), key=os.path.getctime)
        return files

    def plot_confusion_matrix(y_true, y_pred, multilabels, normalize=True, labels=None, title=None, verbose=0):
        """
        Plots a confusion matrix.
        
        Keyword arguments:
        y_true -- numpy array of ground truth labels
        y_pred -- numpy array of predicted labels
        multilabels -- True if the labels are multi-labels, False if they are mono-labels
        normalize -- True if the confusion matrix should be normalized, False otherwise (default: True)
        labels -- list of unique labels that may appear in y_true and y_pred (default: None)
        title -- title of the plot (default: None)
        verbose -- degree of method's verbosity, if set to 2, plots the conf matrix (default: 0)
        
        Return: 
        1. Matplotlib Figure containing the confusion matrix plot
        2. Matplotlib Axes used to plot the confusion matrix
        3. Numpy nd array. Confusion matrix whose i-th row and j-th column entry indicates the number of samples
           with true label being i-th class and predicted label being j-th class
        """
        if labels is None and multilabels is True:
            raise Exception('Labels list must be specified if multilabels is True')
        # elif labels is not None and multilabels is False:
        #     warnings.warn("Warning: given labels list won't be used for confusion matrix plot.")
            
        if verbose < 2:
            plt.ioff()
            
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
            
        normalize = 'all' if normalize else None
        if multilabels:
            cm = multilabel_confusion_matrix(y_true, y_pred)
            if normalize == 'all':
                cm = cm / cm.sum()
            labels = list(labels.keys())
        else:
            cm = confusion_matrix(y_true, y_pred, normalize=normalize)
            
        ncols = 3 if multilabels else 1
        nrows = math.ceil(len(labels) / ncols) if multilabels else 1
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize = (14, 14) if multilabels else (7, 7), dpi=80)
        if not issubclass(type(axes), matplotlib.axes.SubplotBase) and \
            (type(axes[0]) is list or type(axes[0]) is np.array or type(axes[0]) is np.ndarray):
            axes = list(chain.from_iterable(axes))
        fig.suptitle(title, fontsize=12)
        
        if multilabels:
            for i, (ax, lbl) in enumerate(zip(axes, labels)):
                disp = ConfusionMatrixDisplay(confusion_matrix=cm[i], display_labels=[lbl, 'other'])
                disp.plot(include_values=True, cmap=plt.cm.Blues, ax=ax, colorbar=False)
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels(y_true, y_pred))
            disp.plot(include_values=True, cmap=plt.cm.Blues, ax=axes, colorbar=False)
        fig.tight_layout()
        if verbose >= 2:
            plt.show()
        return fig, axes, cm

    class catchtime(object):
        def __init__(self, title, verbose=True):
            self.title = title
            self.verbose = verbose

        def __enter__(self):
            self.start = perf_counter()
            self.end = None
            return self

        def __exit__(self, type, value, traceback):
            self.end = perf_counter()
            if self.verbose:
                time = {'seconds': self.end - self.start, 'minutes': None, 'hours': None, 'days': None}
                if time['seconds'] > 60 * 10:
                    time['minutes'] = time['seconds'] / 60
                    if time['minutes'] > 60 * 10:
                        time['hours'] = time['minutes'] / 60
                        if time['hours'] > 24 * 3:
                            time['days'] = time['hours'] / 24
                for k, v in list(time.items())[::-1]:
                    if v is not None:
                        display = v, k
                        break
                print('%s: %.2f %s' % (self.title, *display))