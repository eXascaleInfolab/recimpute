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

if __name__ == '__main__':
    print(str(sys.argv))

    datasets = Dataset.instantiate_all_realworld()
    
    for ds in datasets:
        print(ds.name)
        #ds = ds.load_realworld()
        #print(ds.head(2).to_markdown())