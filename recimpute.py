"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
recimpute.py
@author: @chacungu
"""

#!/usr/bin/env python

import numpy as np
import os
from os.path import normpath as normp
import pandas as pd
import re
import sys
import warnings

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
from Utils.Utils import Utils


SYSTEM_INPUTS_DIR = normp('./Datasets/SystemInputs/')
SYSTEM_OUTPUTS_DIR = normp('./Datasets/Recommendations/')

# create necessary directories if not there yet
Utils.create_dirs_if_not_exist([SYSTEM_INPUTS_DIR])
Utils.create_dirs_if_not_exist([SYSTEM_OUTPUTS_DIR])


def train_and_eval():

    print('#########  RecImpute - train & eval  #########')

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
        KiviatFeaturesExtracter.get_instance(),
        TSFreshFeaturesExtracter.get_instance(),
    ]

    if any(isinstance(fe, KiviatFeaturesExtracter) for fe in features_extracters):
        warnings.warn('You are using a KiviatFeaturesExtracter. This features extracter can only compute features for clusters' \
                    + '(and not individual time series). If you use the resulting models in production, since those time series' \
                    + 'won\'t be clustered, its features will have to be imputed (or set to 0). This may impact the system\'s' \
                    + 'performances.')

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

    print('=================== Cross-validation results (averaged) ===================')
    print(tr.results[['Accuracy', 'F1-Score', 'Precision', 'Recall']].to_markdown())

    return tr, set, models

def use(timeseries, model, features_name, fes_names, use_pipeline_prod=True):

    print('#########  RecImpute - use a model  #########')

    # get an instance of each FeaturesExtracter
    features_extracters = []
    for fe_name in fes_names:
        if fe_name != 'KiviatFeaturesExtracter':
            fe_class = getattr(sys.modules[__name__], fe_name)
            features_extracters.append(fe_class.get_instance())

    # for each FeaturesExtracter: call fe.extract_from_timeseries()
    nb_timeseries, timeseries_length = timeseries.shape
    all_ts_features = []
    for features_extracter in features_extracters:
        tmp_ts_features = features_extracter.extract_from_timeseries(timeseries.T, nb_timeseries, timeseries_length)
        tmp_ts_features.set_index('Time Series ID', inplace=True)
        all_ts_features.append(tmp_ts_features)
    timeseries_features = pd.concat(all_ts_features, axis=1) # concat features dataframes

    # remove unwanted features (those not listed in features_name)
    timeseries_features = timeseries_features.loc[:, timeseries_features.columns.isin(features_name)]

    if not (list(timeseries_features.columns) == list(features_name)):
        # some features were not computed for the new data: impute those missing values
        missing_features_l = list(set(features_name) - set(timeseries_features.columns))
        missing_features = dict(zip(missing_features_l, [list(features_name).index(f) for f in missing_features_l]))
        for feature, feature_index in dict(sorted(missing_features.items(), key=lambda item: item[1])).items():
            # TODO: find a clever way to impute missing feature values (maybe store the avg in the models?)
            imputed_feature_values = np.zeros(nb_timeseries)
            timeseries_features.insert(feature_index, feature, imputed_feature_values)
        perc_missing_features = len(missing_features_l) / len(features_name)

        warning_text = '/!\ An important number of features' if perc_missing_features > 0.20 else 'Some feature(s)'
        warning_text += ' (%i) could not be computed and their values have been set to 0.' % len(missing_features_l) \
                      + ' This may impact the system\'s performances.' 
        warnings.warn(warning_text)

    # verify the features are the same and in the same order as when the system was trained
    assert list(timeseries_features.columns) == list(features_name)

    # use the model to get recommendation(s) for each time series
    X = timeseries_features.to_numpy().astype('float32')
    labels = model.predict(X, use_pipeline_prod=use_pipeline_prod)
    
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

def get_recommendations_filename(timeseries_filename):
    labels_filename = '.'.join(timeseries_filename.split('.')[:-1]) + '__recommendations.csv'
    labels_filename = normp(SYSTEM_OUTPUTS_DIR + '/' + labels_filename)
    return labels_filename


if __name__ == '__main__':
    print(str(sys.argv))

    CMD = 'USE' # TODO: this should be a parameter

    if CMD == 'TRAIN': # TRAIN and EVALUATE W/ CROSS-VAL
        tr, set, models = train_and_eval()
        print(tr.id)
    
    elif CMD == 'USE': # USE PRE-TRAINED MODELS
        # load the model
        id = '0411_1456_53480' # TODO: this should be a parameter
        model_name = 'kneighbors' # TODO: this should be a parameter
        tr, model = load_models_from_tr(id, model_name)

        # load time series to label: z-normalized, 1 row = 1 ts, space separator, no header, no index
        ts_filename = 'timeseries.csv' # TODO: this should be a parameter
        full_ts_filename = normp(SYSTEM_INPUTS_DIR + '/' + ts_filename)
        timeseries = pd.read_csv(full_ts_filename, sep=' ', header=None, index_col=None)

        # read the _info.txt and get the values under "## Features extracters used:"
        info_file = tr.get_info_file()
        fes_names = re.search('## Features extracters used:\n(- \w+\n)+', info_file).group(0).replace('- ', '').split('\n')[1:-1]

        # get the recommendations
        use_pipeline_prod = False # TODO: this should be a parameter
        labels = use(timeseries, model, model.features_name, fes_names, use_pipeline_prod=use_pipeline_prod)

        print('============================= Recommendations =============================')
        print(labels)

        # save the recommendations to disk
        labels_df = pd.DataFrame(labels, columns=['Recommendations'])
        labels_df.index.name = 'Time Series ID'
        labels_df.to_csv(get_recommendations_filename(ts_filename), sep=' ', header=True, index=True)