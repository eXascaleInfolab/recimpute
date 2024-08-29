import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('../')

from recimpute import select
from Utils.Utils import Utils

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from tqdm import tqdm

class SimpleTrainingSet():
    # Class that imits the TrainingSet class but allows us to create random time series
    def __init__(self, nb_timeseries, ts_len=10, nb_labels=2):
        self.properties = {
            'type': 'monolabels',
        }
        data = np.random.rand(nb_timeseries, ts_len)
        self.data = pd.DataFrame(data)
        self.data.index.name = 'New Time Series ID'
        self.labels = pd.Series((np.random.rand(nb_timeseries) * nb_labels).astype(int))
        self.labels.index.name = 'New Time Series ID'
        self.unique_labels = self.labels.unique().astype(str)

    def get_labeler_properties(self):
        return self.properties

    def get_default_properties(self):
        return None

    def yield_splitted_train_val(self, data_properties, nb_cv_splits):
        for cv_split in range(nb_cv_splits):
            X_train, X_val, y_train, y_val = sklearn_train_test_split(
                self.data, self.labels, shuffle=True, test_size=0.3,
            )
            yield self.data, self.labels, self.unique_labels, X_train, y_train.astype(str), X_val, y_val.astype(str)

def main():

    MODELSTRAINER_CONF = Utils.read_conf_file('modelstrainer')
    params = [ # Warning: please use enough time series to have enough time series per class even after splitting into train/val/test
        # (nb_timeseries, length of time series)
        (  100          ,  10 ), 
        (  1_000        ,  10 ), 
        (  10_000       ,  10 ), 
        (  100_000      ,  10 )
    ]

    # vary nb of time series in the training set
    for nb_timeseries, ts_len in tqdm(params):
        print('~~ Starting selection of pipelines with %i time series of length %i ~~' % (nb_timeseries, ts_len))
        training_set = SimpleTrainingSet(nb_timeseries=nb_timeseries,
                                         ts_len=ts_len)
        with Utils.catchtime('ModelRace time', verbose=True):
            select(training_set, MODELSTRAINER_CONF)
    print('Done')

if __name__ == '__main__':
    main()