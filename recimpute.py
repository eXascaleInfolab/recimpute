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
from Clustering.ShapeBasedClustering import ShapeBasedClustering
from Labeling.ImputationTechniques.ImputeBenchLabeler import ImputeBenchLabeler


if __name__ == '__main__':
    print(str(sys.argv))

    datasets = Dataset.instantiate_from_dir(cassignment_created=True, labels_created=False)

    # if labels_created is set to True we can do:
    # for ds in datasets:
    #   ds.set_labeler(ImputeBenchLabeler)
    # and then one can access the labels with datasets[0].load_labels(properties)
    
    for ds in datasets:
        pass
        #print(ds.name)
        #timeseries = ds.load_timeseries()
        #print(timeseries.head(2).to_markdown())

    #cm = ShapeBasedClustering()
    #datasets = cm.run(datasets)

    ibl = ImputeBenchLabeler()
    datasets = ibl.label(datasets)