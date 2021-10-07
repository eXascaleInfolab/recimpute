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
from Datasets.TrainingSet import TrainingSet
from Clustering.ShapeBasedClustering import ShapeBasedClustering
from Labeling.ImputationTechniques.ImputeBenchLabeler import ImputeBenchLabeler
from Labeling.ImputationTechniques.KiviatRulesLabeler import KiviatRulesLabeler
from FeaturesExtraction.KiviatFeaturesExtracter import KiviatFeaturesExtracter
from FeaturesExtraction.TSFreshFeaturesExtracter import TSFreshFeaturesExtracter


def run_all():
    datasets = Dataset.instantiate_from_dir()
    
    clusterer = ShapeBasedClustering()
    datasets = clusterer.cluster_all_datasets(datasets)

    labeler = ImputeBenchLabeler.get_instance() # KiviatRulesLabeler
    datasets = labeler.label(datasets)

    # TODO

def tests():
    datasets = Dataset.instantiate_from_dir()[:2]

    labeler = ImputeBenchLabeler.get_instance() # KiviatRulesLabeler

    properties = labeler.get_labels_possible_properties()
    properties['type'] = 'multilabels'
    properties['multi_labels_nb_rel'] = 3
    properties['reduction_threshold'] = .05

    labels, labels_list = datasets[0].load_labels(labeler, properties)
    print(datasets[0].name)
    print(labels_list)
    print(labels.to_markdown())


if __name__ == '__main__':
    print(str(sys.argv))

    #run_all()
    #tests()

    #datasets = Dataset.instantiate_from_dir()[:3]

    datasets = [Dataset('ACSF1.zip'), Dataset('Adiac.zip'),Dataset('Beef.zip')]
    clusterer = ShapeBasedClustering()
    labeler = KiviatRulesLabeler.get_instance()
    true_labeler = ImputeBenchLabeler.get_instance()
    features_extracters = [
        KiviatFeaturesExtracter.get_instance(),
        TSFreshFeaturesExtracter.get_instance(),
    ]

    set = TrainingSet(datasets, clusterer, features_extracters, labeler, true_labeler, force_generation=False)