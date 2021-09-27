"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
recimpute.py
@author: @chacungu
"""

#!/usr/bin/env python

import sys

from Datasets.Dataset import Dataset
from Clustering.ClusterModel import ClusterModel


if __name__ == '__main__':
    print(str(sys.argv))

    datasets = Dataset.instantiate_from_dir(cassignment_created=False)
    
    for ds in datasets:
        pass
        #print(ds.name)
        #timeseries = ds.load_timeseries()
        #print(timeseries.head(2).to_markdown())

    cm = ClusterModel()
    datasets = cm.run(datasets)