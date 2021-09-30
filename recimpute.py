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


def run_all():
    datasets = Dataset.instantiate_from_dir(cassignment_created=False, labels_created=False)
    
    cm = ShapeBasedClustering()
    datasets = cm.run(datasets)

    ibl = ImputeBenchLabeler()
    datasets = ibl.label(datasets)

    # TODO

def tests():
    datasets = Dataset.instantiate_from_dir(cassignment_created=True, labels_created=True)

    for ds in datasets:
        ds.set_labeler(ImputeBenchLabeler)
    
    properties = ImputeBenchLabeler.get_labels_possible_properties()
    properties['type'] = 'multilabels'
    properties['multi_labels_nb_rel'] = 3
    properties['reduction_threshold'] = .05

    labels, labels_list = datasets[0].load_labels(properties)
    print(labels_list)
    print(labels.to_markdown())


if __name__ == '__main__':
    print(str(sys.argv))

    tests()