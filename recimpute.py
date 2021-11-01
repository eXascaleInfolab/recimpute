"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
recimpute.py
@author: @chacungu
"""

#!/usr/bin/env python

import sys

from Clustering.ShapeBasedClustering import ShapeBasedClustering
from Datasets.Dataset import Dataset
from Datasets.TrainingSet import TrainingSet
from Evaluation.ModelsEvaluater import ModelsEvaluater
from FeaturesExtraction.KiviatFeaturesExtracter import KiviatFeaturesExtracter
from FeaturesExtraction.TSFreshFeaturesExtracter import TSFreshFeaturesExtracter
from Labeling.ImputationTechniques.ImputeBenchLabeler import ImputeBenchLabeler
from Labeling.ImputationTechniques.KiviatRulesLabeler import KiviatRulesLabeler
from Training.ModelsTrainer import ModelsTrainer
from Training.RecommendationModel import RecommendationModel
from Training.TrainResults import TrainResults


def run_all():

    print('#########  RecImpute - run all  #########')

    # init data sets
    datasets = Dataset.instantiate_from_dir()
    print('Loaded data sets:', ''.join(['\n- %s' % d for d in datasets]))

    # clustering
    clusterer = ShapeBasedClustering()

    # labeling
    labeler = ImputeBenchLabeler.get_instance() #KiviatRulesLabeler.get_instance()
    #true_labeler = ImputeBenchLabeler.get_instance()
    labeler_properties = labeler.get_default_properties()
    #true_labeler_properties = true_labeler.get_default_properties()
    #true_labeler_properties['type'] = labeler_properties['type']

    # features extraction
    features_extracters = [
        KiviatFeaturesExtracter.get_instance(),
        TSFreshFeaturesExtracter.get_instance(),
    ]

    # create a training set
    set = TrainingSet(
        datasets, 
        clusterer, 
        features_extracters, 
        labeler, labeler_properties,
        #true_labeler=true_labeler, true_labeler_properties=true_labeler_properties,
        force_generation=False,
    )

    # init models
    models_descriptions_to_use = [
        'kneighbors.py',
        'maxabsscaler_catboostclassifier.py',
        'normalizer_randomforest.py',
        'standardscaler_randomforest.py',
        #'standardscaler_svc.py',
    ]
    models = RecommendationModel.init_from_descriptions(models_descriptions_to_use)

    # training
    trainer = ModelsTrainer(set, models)
    tr = trainer.train()

    print("=======================================") # TODO tmp
    print(tr.results[['Accuracy', 'F1-Score', 'Precision', 'Recall']].to_markdown()) # TODO tmp

    # testing
    evaluater = ModelsEvaluater(set, models, tr)
    evaluater.test_models()

    return tr, set, models


if __name__ == '__main__':
    print(str(sys.argv))

    tr, set, models = run_all()
    id = tr.id
    print(id)


    """ TODO
    Save & load prev. training results & trained models:
    - pickle dump the RecommendationModels
        - OK: add labels_info
        - OK: add labels_set
        - OK: add features_name
        - add trained_pipeline - save the best model out of the training? - questions
        - OK: dump all except trained_pipeline
        - OK: dump trained_pipeline independently
        - load & initialization methods
        - fix ModelsEvaluater
    """

    

    # ---

    # id = '1410_1546_53480'
    # print(id)
    # tr = TrainResults.load(id)