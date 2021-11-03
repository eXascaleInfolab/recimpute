"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
recimpute.py
@author: @chacungu
"""

#!/usr/bin/env python

import re
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

    # labeling # TODO: the choice of which labeler & true_labeler use should be a parameter
    labeler = ImputeBenchLabeler.get_instance() #KiviatRulesLabeler.get_instance()
    #true_labeler = ImputeBenchLabeler.get_instance()
    labeler_properties = labeler.get_default_properties()
    #true_labeler_properties = true_labeler.get_default_properties()
    #true_labeler_properties['type'] = labeler_properties['type']

    # features extraction # TODO: the choice of which feature extracter use should be a parameter
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

    # init models # TODO: the choice of which model train should be a parameter
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
    train_on_all_data = False # TODO: this should be a parameter
    tr = trainer.train(train_on_all_data=train_on_all_data) 

    print("=================== Cross-validation results (averaged) ===================")
    print(tr.results[['Accuracy', 'F1-Score', 'Precision', 'Recall']].to_markdown())

    return tr, set, models

def use(timeseries, model, features_name, fes_names, use_pipeline_prod=True):

    print('#########  RecImpute - use a model  #########')

    # get an instance of each FeaturesExtracter
    features_extracters = []
    for fe_name in fes_names:
        fe_class = getattr(sys.modules[__name__], fe_name)
        features_extracters.append(fe_class.get_instance())

    # for each FeaturesExtracter: call fe.extract_from_timeseries()
    nb_timeseries, timeseries_length = timeseries.shape()
    all_ts_features = []
    for features_extracter in self.features_extracters:
        tmp_ts_features = features_extracter.extract_from_timeseries(timeseries, nb_timeseries, timeseries_length)
        tmp_ts_features.set_index('Time Series ID', inplace=True)
        all_ts_features.append(tmp_ts_features)
    timeseries_features = pd.concat(all_ts_features, axis=1) # concat features dataframes

    # remove unwanted features (those not listed in features_name)
    timeseries_features = timeseries_features.loc[:, timeseries_features.columns.isin(features_name)]

    # verify that the features' order is the same as in features_name and that all features are there
    assert timeseries_features.columns == features_name

    # use the model to get recommendation(s) for each time series
    X = timeseries_features.to_numpy().astype('float32')
    labels = model.predict(X, use_pipeline_prod=True)
    
    return labels

def load_models_from_tr(id, model_names):
    tr = TrainResults.load(id)
    single_model = False
    if type(model_names) == str:
        single_model = True
        model_names = [model_names]
    selected_models = []
    for model in tr.models:
        if model.name in model_names:
            selected_models.append(model)
    assert len(model_names) == len(selected_models)
    return tr, selected_models[0] if single_model else selected_models


if __name__ == '__main__':
    print(str(sys.argv))

    CMD = 'RUN' # TODO: this should be a parameter

    if CMD == 'RUN': # RUN and EVALUATE W/ CROSS-VAL
        tr, set, models = run_all()
        print(tr.id)
    
    elif CMD == 'USE': # USE MODELS
        id = '0311_1541_53480' # TODO: this should be a parameter
        model_name = 'kneighbors' # TODO: this should be a parameter
        tr, model = load_models_from_tr(id, model_name)
        timeseries = None # TODO: load the timeseries we want to label

        # read the _info.txt and get the values under "## Features extracters used:"
        info_file = tr.get_info_file()
        fes_names = re.search('## Features extracters used:\n(- \w+\n)+', info_file).group(0).replace('- ', '').split('\n')[1:-1]
        labels = use(timeseries, model, model.features_name, fes_names)
        print(labels) # TODO should the predicted labels be stored to disk? probably