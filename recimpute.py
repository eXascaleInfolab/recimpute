"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
recimpute.py
@author: @chacungu
"""

#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import normpath as normp
import pandas as pd
from pprint import pprint
import re
import sys
import warnings

from Clustering.ShapeBasedClustering import ShapeBasedClustering
from Datasets.Dataset import Dataset
from Datasets.TrainingSet import TrainingSet
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

LABELERS = { # maps the argument name to the actual class name
    'ImputeBench': ImputeBenchLabeler, 
    'KiviatRules': KiviatRulesLabeler,
}
FEATURES_EXTRACTERS = { # maps the argument name to the actual class name
    'Kiviat': KiviatFeaturesExtracter, 
    'TSFresh': TSFreshFeaturesExtracter,
}


# create necessary directories if not there yet
Utils.create_dirs_if_not_exist([SYSTEM_INPUTS_DIR])
Utils.create_dirs_if_not_exist([SYSTEM_OUTPUTS_DIR])


def train(labeler, labeler_properties, true_labeler, true_labeler_properties, features_extracters, models_descriptions_to_use, train_on_all_data):

    print('#########  RecImpute - training  #########')

    # init clusterer
    clusterer = ShapeBasedClustering()

    # init data sets
    datasets = Dataset.instantiate_from_dir(clusterer)
    print('Loaded data sets:', ''.join(['\n- %s' % d for d in datasets]))

    if any(isinstance(fe, KiviatFeaturesExtracter) for fe in features_extracters):
        warnings.warn('You are using a KiviatFeaturesExtracter. This features extracter can only compute features for clusters' \
                    + '(and not individual time series). If you use the resulting models in production, since those time series' \
                    + 'won\'t be clustered, its features will have to be imputed (or set to 0). This may impact the system\'s' \
                    + 'performances.')

    true_labeler_info = {'true_labeler': true_labeler, 'true_labeler_properties': true_labeler_properties} \
                        if true_labeler is not None else {}

    # create a training set
    training_set = TrainingSet(
        datasets, 
        clusterer, 
        features_extracters, 
        labeler, labeler_properties,
        **true_labeler_info,
        force_generation=False,
    )

    models = RecommendationModel.init_from_descriptions(models_descriptions_to_use)

    # training & cross-validation evaluation
    trainer = ModelsTrainer(training_set, models)
    tr = trainer.train(train_on_all_data=train_on_all_data) 

    print('\n\n=================== Cross-validation results (averaged) ===================')
    print(tr.results[tr.metrics_measured].to_markdown())

    return tr, training_set, models

def eval(models, all_test_data_info):
    
    print('#########  RecImpute - evaluation  #########')

    X_test = all_test_data_info.iloc[:, ~all_test_data_info.columns.isin(['Cluster ID', 'Label'])]
    y_test = all_test_data_info['Label']

    for model in models:
        print(model)
        used_tp, y_pred = model.predict(X_test, compute_proba=model.labels_info['type']=='monolabels', use_pipeline_prod=False)
        scores, cm = model.eval(y_test, y_pred, used_tp.classes_, plot_cm=True)

        print('\n# %s' % model.name)
        pprint(scores, width=1)
        print(np.array_str(cm[1], precision=3, suppress_small=False))

        fig = cm[0]
        fig.canvas.draw()
        renderer = fig.canvas.renderer
        fig.draw(renderer)

def use(timeseries, model, features_name, fes_names, use_pipeline_prod=True):

    print('#########  RecImpute - use a model  #########')

    # get an instance of each FeaturesExtracter
    features_extracters = []
    for fe_name in fes_names:
        if fe_name != 'KiviatFeaturesExtracter':
            assert any(fe_name2.__name__ == fe_name for fe_name2 in FEATURES_EXTRACTERS.values())
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
    recommendations = model.get_recommendations(X, use_pipeline_prod=use_pipeline_prod)
    
    return recommendations

def load_models_from_tr(id, model_names=None):
    # loads all models if model_names is None
    # otherwise loads the models which name is listed in model_names
    tr = TrainResults.load(id)
    single_model = False
    if type(model_names) == str:
        single_model = True
        model_names = [model_names]
    selected_models = []
    for model in tr.models:
        if model_names is None or model.name in model_names:
            selected_models.append(model)
    assert model_names is None or len(model_names) == len(selected_models)
    return tr, selected_models[0] if single_model else selected_models

def get_recommendations_filename(timeseries_filename):
    labels_filename = '.'.join(timeseries_filename.split('.')[:-1]) + '__recommendations.csv'
    labels_filename = normp(SYSTEM_OUTPUTS_DIR + '/' + labels_filename)
    return labels_filename


# --------------------------------------------------------------------------------------------



if __name__ == '__main__':
    
    _models_list = [f.replace('.py', '') for f in os.listdir('Training/ModelsDescription') if f not in ['__pycache__', '_template.py']]
    _valid_args = {
        '-mode': ['train', 'eval', 'use'],

        # *train* args
        '-lbl': LABELERS.keys(),
        '-true_lbl': LABELERS.keys(),
        '-fes': [*FEATURES_EXTRACTERS.keys(), 'all'],
        '-models': [*_models_list, 'all'],
        '-train_on_all_data': ['True', 'False'],

        # *eval* args
        '-id': None,

        # *use* args
        '-id': None,
        '-model': _models_list,
        '-ts': None,
        '-use_prod_model': ['True', 'False'],
    }

    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    assert '-mode' in args and args['-mode'] in _valid_args['-mode'] # verify the -mode arg has been specified correctly
    assert all(k in _valid_args.keys() for k in args.keys()) # verify that all args keys are valid
    assert all(_valid_args[k] is None \
           or (v in _valid_args[k] if not ',' in v else (v_ in _valid_args[k] for v_ in v.split(','))) \
              for k, v in args.items()) # verify that all args values are valid
    
    if args['-mode'] == 'train':

        NON_OPTIONAL_ARGS = ['-mode', '-lbl', '-fes', '-models']
        assert all(noa in args.keys() for noa in NON_OPTIONAL_ARGS) # verify that all non-optional args are specified
        
        # TRAIN and EVALUATE W/ CROSS-VAL
        
        # set up the labeler & true_labeler
        labeler = LABELERS[args['-lbl']].get_instance()
        labeler_properties = labeler.get_default_properties()

        if '-true_lbl' in args:
            true_labeler = LABELERS[args['-true_lbl']].get_instance()
            true_labeler_properties = true_labeler.get_default_properties()
            true_labeler_properties['type'] = labeler_properties['type']
        else:
            true_labeler = true_labeler_properties = None

        # set up the features extracters
        if args['-fes'] == 'all':
            features_extracters = [fe.get_instance() for fe in FEATURES_EXTRACTERS.values()]
        else:
            features_extracters = []
            for fe_name in args['-fes'].split(','):
                features_extracters.append(FEATURES_EXTRACTERS[fe_name].get_instance())
        
        # set up the models' descriptions to load
        if args['-models'] == 'all':
            models_descriptions_to_use = [m + '.py' for m in _models_list]
        else:
            models_descriptions_to_use = []
            for m_name in args['-models'].split(','):
                models_descriptions_to_use.append(m_name + '.py')


        tr, set, models = train(
            labeler, labeler_properties, 
            true_labeler, true_labeler_properties, 
            features_extracters, 
            models_descriptions_to_use, 
            args['-train_on_all_data'] == 'True' if '-train_on_all_data' in args else True
        )
        print(tr.id)


    elif args['-mode'] == 'eval':

        NON_OPTIONAL_ARGS = ['-id']
        assert all(noa in args.keys() for noa in NON_OPTIONAL_ARGS) # verify that all non-optional args are specified

        # load the models & test set
        id = args['-id']
        tr, models = load_models_from_tr(id)
        all_test_data_info = tr.load_test_set_from_archive()
        
        eval(models, all_test_data_info)


    elif args['-mode'] == 'use':

        NON_OPTIONAL_ARGS = ['-mode', '-id', '-model', '-ts']
        assert all(noa in args.keys() for noa in NON_OPTIONAL_ARGS) # verify that all non-optional args are specified

        # USE PRE-TRAINED MODELS

        # load the model
        id = args['-id']
        model_name = args['-model']
        tr, model = load_models_from_tr(id, model_name)

        # load time series to label: z-normalized, 1 row = 1 ts, space separator, no header, no index
        ts_filename = args['-ts']
        full_ts_filename = normp(SYSTEM_INPUTS_DIR + '/' + ts_filename)
        timeseries = pd.read_csv(full_ts_filename, sep=' ', header=None, index_col=None)

        # read the _info.txt and get the values under "## Features extracters used:"
        info_file = tr.load_info_file_from_archive()
        fes_names = re.search('## Features extracters used:\n(- \w+\n)+', info_file).group(0).replace('- ', '').split('\n')[1:-1]

        use_pipeline_prod = args['-use_prod_model'] == 'True' if '-use_prod_model' in args else False

        # get the recommendations
        preds = use(timeseries, model, model.features_name, fes_names, use_pipeline_prod=use_pipeline_prod)

        print('============================= Recommendations =============================')
        print(preds)

        # save the recommendations to disk
        preds.to_csv(get_recommendations_filename(ts_filename), sep=' ', header=True, index=True)