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
import scipy
import statsmodels
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

def init_training_set(labeler, labeler_properties, true_labeler, true_labeler_properties, features_extractors):
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
    return training_set

def select(training_set, MODELSTRAINER_CONF):

    print('#########  RecImpute - models\' selection  #########')

    significance_tests = {
        'ttest_rel': scipy.stats.ttest_rel,
        'friedmanchisquare': scipy.stats.friedmanchisquare,
        'chisquare': scipy.stats.chisquare,
        'ztest': statsmodels.stats.weightstats.ztest,
    }

    pipelines, all_pipelines_txt = ClfPipeline.generate(N=MODELSTRAINER_CONF['NB_PIPELINES'])

    # most promising pipelines' selection
    trainer = ModelsTrainer(training_set)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        selected_pipes = trainer.select(
            pipelines, all_pipelines_txt, 
            S=MODELSTRAINER_CONF['S'], 
            selection_len=MODELSTRAINER_CONF['SELECTION_LEN'], 
            score_margin=MODELSTRAINER_CONF['SCORE_MARGIN'],
            n_splits=MODELSTRAINER_CONF['NB_CV_SPLITS'], 
            test_method=significance_tests[MODELSTRAINER_CONF['TEST_METHOD']], 
            p_value=MODELSTRAINER_CONF['P_VALUE'],
            alpha=MODELSTRAINER_CONF['ALPHA'],
            beta=MODELSTRAINER_CONF['BETA'],
            gamma=MODELSTRAINER_CONF['GAMMA'],
            allow_early_eliminations=MODELSTRAINER_CONF['ALLOW_EARLY_ELIMINATIONS'],
            early_break=False,
        )

    print('Finished models\' selection with %i remaining candidates.' % len(selected_pipes)) # TODO tmp print
    
    models = list(map(lambda p: p.rm, selected_pipes))
    return models

def train(models, training_set, train_on_all_data):
    print('#########  RecImpute - training  #########')

    # training & cross-validation evaluation
    trainer = ModelsTrainer(training_set)
    tr = trainer.train(models, train_on_all_data=train_on_all_data) 

    print('\n\n=================== Cross-validation results (averaged) ===================')
    print(tr.results[tr.metrics_measured].to_markdown())

    return tr, training_set, models

def eval(models, all_test_data_info, print_details=False):
    
    print('#########  RecImpute - evaluation  #########')

    X_test = all_test_data_info.iloc[:, ~all_test_data_info.columns.isin(['Data Set Name', 'Cluster ID', 'Label'])].to_numpy().astype('float32')
    X_test = np.nan_to_num(X_test)
    y_test = all_test_data_info['Label'].to_numpy().astype('str')

    DATASETS_CONF = Utils.read_conf_file('datasets')
    categories = np.array(list(map(lambda ds_name: DATASETS_CONF['CATEGORIES'][ds_name], all_test_data_info['Data Set Name'].tolist())))

    all_scores, all_scores_per_category = {}, {}
    for model in models:
        if print_details:
            print(model)
        used_tp, y_pred = model.predict(X_test, compute_proba=model.labels_info['type']=='monolabels', use_pipeline_prod=False)
        scores, cm, scores_per_category = model.eval(y_test, y_pred, used_tp.classes_, categories=categories, plot_cm=True)

        for k,v in scores.items():
            if k in all_scores:
                all_scores[k].append(v)
            else:
                all_scores[k] = [v]
        for category in scores_per_category.keys():
            if category not in all_scores_per_category:
                all_scores_per_category[category] = {}
            for k,v in scores_per_category[category][0].items():
                if k in all_scores_per_category[category]:
                    all_scores_per_category[category][k].append(v)
                else:
                    all_scores_per_category[category][k] = [v]

        if print_details:
            print('\n# %s - %s' % (model.id, model.pipe))
            pprint(scores, width=1)
            print(np.array_str(cm[1], precision=3, suppress_small=False))

            pprint(scores_per_category, width=1)

            fig = cm[0]
            fig.canvas.draw()
            renderer = fig.canvas.renderer
            fig.draw(renderer)
    
    print("***")
    print(all_scores_per_category)
    print('\nAverage results:')
    pprint(dict(map(lambda i: (i[0], np.mean(i[1])), all_scores.items())))
    pprint({k1: {k2: np.mean(v2) for k2,v2 in v1.items()} for k1,v1 in all_scores_per_category.items()})

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
        args = (timeseries.T, nb_timeseries, timeseries_length) if isinstance(features_extractor, TSFreshFeaturesExtractor) else (timeseries.T,)
        tmp_ts_features = features_extractor.extract_from_timeseries(*args)
        tmp_ts_features.set_index('Time Series ID', inplace=True)
        tmp_ts_features.columns = map(
            lambda col_name: col_name + features_extractor.FEATURES_FILENAMES_ID if col_name not in ['Time Series ID'] else col_name, 
            tmp_ts_features.columns
        )
        all_ts_features.append(tmp_ts_features)
    timeseries_features = pd.concat(all_ts_features, axis=1) # concat features dataframes

    # remove unwanted features (those not listed in features_name)
    timeseries_features = timeseries_features.loc[:, timeseries_features.columns.isin(features_name)]

    if not (list(timeseries_features.columns) == list(features_name)):
        # some features were not computed for the new data: impute those missing values
        missing_features_l = list(set(features_name) - set(timeseries_features.columns))
        missing_features = dict(zip(missing_features_l, [list(features_name).index(f) for f in missing_features_l]))
        for feature, feature_index in dict(sorted(missing_features.items(), key=lambda item: item[1])).items():
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

def load_models_from_tr(id, model_ids=None):
    # loads all models if model_ids is None
    # otherwise loads the models which id is listed in model_ids
    tr = TrainResults.load(id)
    single_model = False
    if type(model_ids) == str:
        single_model = True
        model_ids = [model_ids]
    selected_models = []
    for model in tr.models:
        if model_ids is None or str(model.id) in model_ids:
            selected_models.append(model)
    assert model_ids is None or len(model_ids) == len(selected_models)
    return tr, selected_models[0] if single_model else selected_models

def get_recommendations_filename(timeseries_filename):
    labels_filename = '.'.join(timeseries_filename.split('.')[:-1]) + '__recommendations.csv'
    labels_filename = normp(SYSTEM_OUTPUTS_DIR + '/' + labels_filename)
    return labels_filename


# --------------------------------------------------------------------------------------------

def main(args):
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
        '-model_id': None,
        '-use_prod_model': ['True', 'False'],
    }

    args = dict(zip(args[1::2], args[2::2]))
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
        
        MODELSTRAINER_CONF = Utils.read_conf_file('modelstrainer')

        training_set = init_training_set(labeler, labeler_properties, true_labeler, true_labeler_properties, features_extractors)
        models = select(training_set, MODELSTRAINER_CONF)
        tr, set, models = train(models, training_set, args['-train_on_all_data'] == 'True' if '-train_on_all_data' in args else True)
        print('Done.')
        print(tr.id)
        return tr, set, models


    elif args['-mode'] == 'eval':

        NON_OPTIONAL_ARGS = ['-id']
        assert all(noa in args.keys() for noa in NON_OPTIONAL_ARGS) # verify that all non-optional args are specified

        # load the models & test set
        id = args['-id']
        tr, models = load_models_from_tr(id)
        all_test_data_info = tr.load_set_from_archive('test')
        
        eval(models, all_test_data_info)
        
        print('Done.')


    elif args['-mode'] == 'use':

        NON_OPTIONAL_ARGS = ['-mode', '-id', '-model_id', '-ts']
        assert all(noa in args.keys() for noa in NON_OPTIONAL_ARGS) # verify that all non-optional args are specified

        # USE PRE-TRAINED MODELS

        # load the model
        id = args['-id']
        model_id = args['-model_id']
        tr, model = load_models_from_tr(id, model_id)

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
        print('Done.')
        return preds

if __name__ == '__main__':

    main(sys.argv)