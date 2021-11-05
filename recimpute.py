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


def train_and_eval(labeler, labeler_properties, true_labeler, true_labeler_properties, features_extracters, models_descriptions_to_use, train_on_all_data):

    print('#########  RecImpute - train & eval  #########')

    # init data sets
    datasets = Dataset.instantiate_from_dir()
    print('Loaded data sets:', ''.join(['\n- %s' % d for d in datasets]))

    # clustering
    clusterer = ShapeBasedClustering()

    if any(isinstance(fe, KiviatFeaturesExtracter) for fe in features_extracters):
        warnings.warn('You are using a KiviatFeaturesExtracter. This features extracter can only compute features for clusters' \
                    + '(and not individual time series). If you use the resulting models in production, since those time series' \
                    + 'won\'t be clustered, its features will have to be imputed (or set to 0). This may impact the system\'s' \
                    + 'performances.')

    true_labeler_info = {'true_labeler': true_labeler, 'true_labeler_properties': true_labeler_properties} \
                        if true_labeler is not None else {}

    # create a training set
    set = TrainingSet(
        datasets, 
        clusterer, 
        features_extracters, 
        labeler, labeler_properties,
        **true_labeler_info,
        force_generation=False,
    )

    models = RecommendationModel.init_from_descriptions(models_descriptions_to_use)

    # training & cross-validation evaluation
    trainer = ModelsTrainer(set, models)
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


# --------------------------------------------------------------------------------------------



if __name__ == '__main__':
    
    _models_list = [f.replace('.py', '') for f in os.listdir('Training/ModelsDescription') if f not in ['__pycache__', '_template.py']]
    _valid_args = {
        '-mode': ['train', 'use'],

        # *train* args
        '-lbl': LABELERS.keys(),
        '-true_lbl': LABELERS.keys(),
        '-fes': [*FEATURES_EXTRACTERS.keys(), 'all'],
        '-models': [*_models_list, 'all'],
        '-train_on_all_data': ['True', 'False'],

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


        tr, set, models = train_and_eval(
            labeler, labeler_properties, 
            true_labeler, true_labeler_properties, 
            features_extracters, 
            models_descriptions_to_use, 
            args['-train_on_all_data'] == 'True' if '-train_on_all_data' in args else True
        )
        print(tr.id)


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
        info_file = tr.get_info_file()
        fes_names = re.search('## Features extracters used:\n(- \w+\n)+', info_file).group(0).replace('- ', '').split('\n')[1:-1]

        use_pipeline_prod = args['-use_prod_model'] == 'True' if '-use_prod_model' in args else False

        print('!! id', id)
        print('!! model_name', model_name)
        print('!! ts_filename', ts_filename)
        print('!! use_pipeline_prod', use_pipeline_prod)

        # get the recommendations
        labels = use(timeseries, model, model.features_name, fes_names, use_pipeline_prod=use_pipeline_prod)

        print('============================= Recommendations =============================')
        print(labels)

        # save the recommendations to disk
        labels_df = pd.DataFrame(labels, columns=['Recommendations'])
        labels_df.index.name = 'Time Series ID'
        labels_df.to_csv(get_recommendations_filename(ts_filename), sep=' ', header=True, index=True)