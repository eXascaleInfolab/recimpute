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
        #KiviatFeaturesExtracter.get_instance(), # TODO: currently not usable if production data is not clustered...
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

    # training & cross-validation evaluation
    trainer = ModelsTrainer(set, models)
    tr = trainer.train()

    print("=======================================") # TODO tmp
    print(tr.results[['Accuracy', 'F1-Score', 'Precision', 'Recall']].to_markdown()) # TODO tmp

    return tr, set, models

def use(timeseries, model, features_name, fes, fes_config): # TODO
    # load config files
    # get an instance of each FeaturesExtracter (one per config file)
    # initialize each FeaturesExtracter with its corresponding loaded config file
    # for each FeaturesExtracter: call fe.extract_from_timeseries()
    # X = concat all features
    # remove unwanted features (those not listed in features_name)
    # verify that the features' order is the same as in features_name and that all features are there
    # y = model.trained_pipeline.predict(X) / predict_proba(X) / more complex recommendation method ?
    # return y
    pass

def load_models_from_tr(id, models_names):
    tr = TrainResults.load(id)
    selected_models = []
    for model in tr.models:
        if model.name in models_names:
            selected_models.append(model)
    return selected_models


if __name__ == '__main__':
    print(str(sys.argv))

    tr, set, models = run_all()
    id = tr.id
    print(id)

    # advanced evaluation in Jupyter Notebooks (TODO)
    # models selection (done by a human)
    # call load_models_from_tr()
    # train them on ALL data before production (TODO) -- problem: TrainingSet needs to be re-created to access the data...
    #                                                    solution: train before the advanced eval -> no more models selection
    # save the final model to disk (TODO)

    # load a final model (TODO)
    # load the time series to label (TODO)
    # call use()

    


    """ TODO
    Save & load prev. training results & trained models:
    - pickle dump the RecommendationModels
        - OK: add labels_info
        - OK: add labels_set
        - OK: add features_name
        - add trained_pipeline - save the best model out of the training? - see questions
        - OK: dump all except trained_pipeline
        - OK: dump trained_pipeline independently
        - OK: load & initialization methods
        - OK: test save & load methods
    - fix ModelsEvaluater (remove and replace by the existing notebooks?)
    """

    

    # ---

    # id = '0211_1723_53480'
    # print(id)
    # tr = TrainResults.load(id)
    # y = tr.models[0].trained_pipeline.predict(X)