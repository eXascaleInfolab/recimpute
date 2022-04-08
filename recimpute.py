"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
recimpute.py
@author: @chacungu
"""

#!/usr/bin/env python

import numpy as np
from os.path import normpath as normp
import pandas as pd
from pprint import pprint
import re
import sys
import warnings

from Clustering.ShapeBasedClustering import ShapeBasedClustering
from Datasets.Dataset import Dataset
from Datasets.TrainingSet import TrainingSet
from FeaturesExtraction.KiviatFeaturesExtractor import KiviatFeaturesExtractor
from FeaturesExtraction.TSFreshFeaturesExtractor import TSFreshFeaturesExtractor
from FeaturesExtraction.TopologicalFeaturesExtractor import TopologicalFeaturesExtractor
from FeaturesExtraction.Catch22FeaturesExtractor import Catch22FeaturesExtractor
from FeaturesExtraction.KatsFeaturesExtractor import KatsFeaturesExtractor
from Labeling.ImputationTechniques.ImputeBenchLabeler import ImputeBenchLabeler
from Labeling.ImputationTechniques.KiviatRulesLabeler import KiviatRulesLabeler
from Training.ClfPipeline import ClfPipeline
from Training.ModelsTrainer import ModelsTrainer
from Training.TrainResults import TrainResults
from Utils.Utils import Utils


SYSTEM_INPUTS_DIR = normp('./Datasets/SystemInputs/')
SYSTEM_OUTPUTS_DIR = normp('./Datasets/Recommendations/')

LABELERS = { # maps the argument name to the actual class name
    'ImputeBench': ImputeBenchLabeler, 
    'KiviatRules': KiviatRulesLabeler,
}
FEATURES_EXTRACTORS = { # maps the argument name to the actual class name
    'Kiviat': KiviatFeaturesExtractor, 
    'TSFresh': TSFreshFeaturesExtractor,
    'Topological': TopologicalFeaturesExtractor,
    'Catch22': Catch22FeaturesExtractor,
    'Kats': KatsFeaturesExtractor,
}


# create necessary directories if not there yet
Utils.create_dirs_if_not_exist([SYSTEM_INPUTS_DIR])
Utils.create_dirs_if_not_exist([SYSTEM_OUTPUTS_DIR])


def train(labeler, labeler_properties, true_labeler, true_labeler_properties, features_extractors, train_on_all_data):

    print('#########  RecImpute - training  #########')

    # init clusterer
    clusterer = ShapeBasedClustering()

    # init data sets
    datasets = Dataset.instantiate_from_dir(clusterer)
    print('Loaded data sets:', ''.join(['\n- %s' % d for d in datasets]))

    if any(isinstance(fe, KiviatFeaturesExtractor) for fe in features_extractors):
        warnings.warn('You are using a KiviatFeaturesExtractor. This features extractor can only compute features for clusters' \
                    + '(and not individual time series). If you use the resulting models in production, since those time series' \
                    + 'won\'t be clustered, its features will have to be imputed (or set to 0). This may impact the system\'s' \
                    + 'performances.')

    true_labeler_info = {'true_labeler': true_labeler, 'true_labeler_properties': true_labeler_properties} \
                        if true_labeler is not None else {}

    # create a training set
    training_set = TrainingSet(
        datasets,
        clusterer, 
        features_extractors, 
        labeler, labeler_properties,
        **true_labeler_info,
        force_generation=False,
    )

    from scipy.stats import ttest_rel#, friedmanchisquare, chisquare
    #from statsmodels.stats.weightstats import ztest as ztest
    nb_pipelines = 300 # TODO
    S = [3, 8, 12, 17, 25] # TODO
    n_splits = 3 # TODO
    test_method = ttest_rel # TODO
    selection_len = 5 # TODO
    score_margin = .2 # TODO
    p_value = .01 # TODO

    pipelines, all_pipelines_txt = ClfPipeline.generate(N=nb_pipelines)

    # most promising pipelines' selection
    trainer = ModelsTrainer(training_set)
    try:
        with warnings.catch_warnings(): # TODO tmp delete
            warnings.simplefilter('ignore') # TODO tmp delete
            selected_pipes = trainer.select(
                pipelines, all_pipelines_txt, 
                S=S, 
                selection_len=selection_len, 
                score_margin=score_margin,
                n_splits=n_splits, 
                test_method=test_method, 
                p_value=p_value,
                early_break=False,
            )
    finally:
        import pickle # TODO tmp delete
        with open('Training/tmp_pipelines.obj', 'wb+') as f: # TODO tmp delete
            pickle.dump(selected_pipes, f) # TODO tmp delete
    #raise Exception('Models selection done.') # TODO delete this line
    
    models = list(map(lambda p: p.rm, selected_pipes))

    # training & cross-validation evaluation
    tr = trainer.train(models, train_on_all_data=train_on_all_data) 

    print('\n\n=================== Cross-validation results (averaged) ===================')
    print(tr.results[tr.metrics_measured].to_markdown())

    return tr, training_set, models

def eval(models, all_test_data_info):
    
    print('#########  RecImpute - evaluation  #########')

    X_test = all_test_data_info.iloc[:, ~all_test_data_info.columns.isin(['Data Set Name', 'Cluster ID', 'Label'])].to_numpy().astype('float32')
    X_test = np.nan_to_num(X_test)
    y_test = all_test_data_info['Label'].to_numpy().astype('str')

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

    # get an instance of each FeaturesExtractor
    features_extractors = []
    for fe_name in fes_names:
        if fe_name != 'KiviatFeaturesExtractor':
            assert any(fe_name2.__name__ == fe_name for fe_name2 in FEATURES_EXTRACTORS.values())
            fe_class = getattr(sys.modules[__name__], fe_name)
            features_extractors.append(fe_class.get_instance())

    # for each FeaturesExtractor: call fe.extract_from_timeseries()
    nb_timeseries, timeseries_length = timeseries.shape
    all_ts_features = []
    for features_extractor in features_extractors:
        tmp_ts_features = features_extractor.extract_from_timeseries(timeseries.T, nb_timeseries, timeseries_length)
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

    _valid_args = {
        '-mode': ['cluster', 'label', 'extract_features', 'train', 'eval', 'use'],

        # *train* args
        '-lbl': LABELERS.keys(),
        '-true_lbl': LABELERS.keys(),
        '-fes': [*FEATURES_EXTRACTORS.keys(), 'all'],
        '-train_on_all_data': ['True', 'False'],

        # *eval* args
        '-id': None,

        # *use* args
        '-id': None,
        '-ts': None,
        '-use_prod_model': ['True', 'False'],
    }

    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    assert '-mode' in args and args['-mode'] in _valid_args['-mode'] # verify the -mode arg has been specified correctly
    assert all(k in _valid_args.keys() for k in args.keys()) # verify that all args keys are valid
    assert all(_valid_args[k] is None \
           or (v in _valid_args[k] if not ',' in v else (v_ in _valid_args[k] for v_ in v.split(','))) \
              for k, v in args.items()) # verify that all args values are valid
    
    if args['-mode'] == 'cluster':

        NON_OPTIONAL_ARGS = ['-mode']
        assert all(noa in args.keys() for noa in NON_OPTIONAL_ARGS) # verify that all non-optional args are specified
        
        # CLUSTER ALL DATA SETS

        # init clusterer
        clusterer = ShapeBasedClustering()

        # init data sets
        datasets = Dataset.instantiate_from_dir(clusterer)

        clusterer.cluster_all_datasets_seq(datasets)
        print('Done.')


    elif args['-mode'] == 'label':

        NON_OPTIONAL_ARGS = ['-mode', '-lbl']
        assert all(noa in args.keys() for noa in NON_OPTIONAL_ARGS) # verify that all non-optional args are specified
        
        # LABEL ALL DATA SETS
        
        # set up the labeler
        labeler = LABELERS[args['-lbl']].get_instance()

        # init clusterer
        clusterer = ShapeBasedClustering()

        # init data sets
        datasets = Dataset.instantiate_from_dir(clusterer)

        # label the datasets' clusters
        updated_datasets = labeler.label_all_datasets(datasets)

        if '-true_lbl' in args:
            true_labeler = LABELERS[args['-true_lbl']].get_instance()
            true_labeler.label_all_datasets(updated_datasets)
        print('Done.')


    elif args['-mode'] == 'extract_features':

        NON_OPTIONAL_ARGS = ['-mode', '-fes']
        assert all(noa in args.keys() for noa in NON_OPTIONAL_ARGS) # verify that all non-optional args are specified
        
        # EXTRACT ALL FEATURES
        
        # set up the features extractors
        if args['-fes'] == 'all':
            features_extractors = [fe.get_instance() for fe in FEATURES_EXTRACTORS.values()]
        else:
            features_extractors = []
            for fe_name in args['-fes'].split(','):
                features_extractors.append(FEATURES_EXTRACTORS[fe_name].get_instance())

        # init clusterer
        clusterer = ShapeBasedClustering()

        # init data sets
        datasets = Dataset.instantiate_from_dir(clusterer)

        # extract the features of the datasets' time series
        for dataset in datasets:
            for features_extractor in features_extractors:
                features_extractor.extract(dataset)
        print('Done.')


    elif args['-mode'] == 'train':

        NON_OPTIONAL_ARGS = ['-mode', '-lbl', '-fes']
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

        # set up the features extractors
        if args['-fes'] == 'all':
            features_extractors = [fe.get_instance() for fe in FEATURES_EXTRACTORS.values()]
        else:
            features_extractors = []
            for fe_name in args['-fes'].split(','):
                features_extractors.append(FEATURES_EXTRACTORS[fe_name].get_instance())
        
        tr, set, models = train(
            labeler, labeler_properties, 
            true_labeler, true_labeler_properties, 
            features_extractors, 
            args['-train_on_all_data'] == 'True' if '-train_on_all_data' in args else True
        )
        print(tr.id)


    elif args['-mode'] == 'eval':

        NON_OPTIONAL_ARGS = ['-id']
        assert all(noa in args.keys() for noa in NON_OPTIONAL_ARGS) # verify that all non-optional args are specified

        # load the models & test set
        id = args['-id']
        tr, models = load_models_from_tr(id)
        all_test_data_info = tr.load_set_from_archive('test')
        
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

        # read the _info.txt and get the values under "## Features extractors used:"
        info_file = tr.load_info_file_from_archive()
        fes_names = re.search('## Features extractors used:\n(- \w+\n)+', info_file).group(0).replace('- ', '').split('\n')[1:-1]

        use_pipeline_prod = args['-use_prod_model'] == 'True' if '-use_prod_model' in args else False

        # get the recommendations
        preds = use(timeseries, model, model.features_name, fes_names, use_pipeline_prod=use_pipeline_prod)

        print('============================= Recommendations =============================')
        print(preds)

        # save the recommendations to disk
        preds.to_csv(get_recommendations_filename(ts_filename), sep=' ', header=True, index=True)