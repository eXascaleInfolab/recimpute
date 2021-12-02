"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
RecommendationModel.py
@author: @chacungu
"""

import importlib.util
from joblib import dump as j_dump, load as j_load
import numpy as np
import os
from os.path import normpath as normp
import pandas as pd
from pickle import dump as p_dump, load as p_load
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, f1_score
from sklearn.pipeline import Pipeline
import time

from Utils.Utils import Utils

class RecommendationModel:
    """
    Class which handles a recommendation model (classification / regression model).
    """

    MODELS_DESCRIPTION_DIR = normp('./Training/ModelsDescription/')
    METRICS_CLF_MONO = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Mean Reciprocal Rank', 'Precision@3', 'Recall@3']
    METRICS_CLF_MULTI = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Hamming Loss']
    METRICS_REGR = ['Mean Squared Error']
    METRIC_FOR_SCORING = {
        'classifier': {'metric': 'F1-Score', 'best_score_is': 'higher'},
        'regression': {'metric': 'Mean Squared Error', 'best_score_is': 'lower'},
    }
    _PROPS_TO_NOT_PICKLE = ['best_cv_trained_pipeline', 'trained_pipeline_prod'] # these are handled separately


    # constructor
    def __init__(self, name, type, steps, params_ranges, training_speed_factor, 
                 multiclass_strategy=None, bagging_strategy=None, description_filename=None, default_params=None):
        """
        Initializes a RecommendationModel object.

        Keyword arguments:
        name -- name of this model
        type -- type of this model (classifier or regression model)
        steps -- list of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they 
                 are chained, with the last object an estimator
        params_ranges -- dict with keys being parameters' names and values being ranges of possible parameter value
        training_speed_factor -- integer between 1 (fast) and 3 (slow) used to indicate the expected training speed of this model
        multiclass_strategy -- Class of a multiclass strategy (default: None)
        bagging_strategy -- Class of a bagging strategy (default: None)
        description_filename -- filename of this model's description file (default: None)
        default_params -- dict of default parameter names mapped to their values. If None and default parameters should be used, the 
                          model's default parameters will be used (default: None)
        """
        self.name = name
        self.type = type
        self.steps_name, self.steps = zip(*steps)
        # for step in self.steps:
        #     assert hasattr(step, 'fit') and callable(getattr(step, 'fit'))
        #     assert hasattr(step, 'predict') and callable(getattr(step, 'predict'))
        #     assert hasattr(step, 'transform') and callable(getattr(step, 'transform'))
        assert type in ['classifier', 'regression']

        self.params_ranges = params_ranges
        self.best_params = None
        self.training_speed_factor = training_speed_factor
        self.multiclass_strategy = multiclass_strategy
        self.bagging_strategy = bagging_strategy
        self.description_filename = description_filename
        self.default_params = default_params

        self.best_cv_trained_pipeline = None
        self.trained_pipeline_prod = None
        self.features_name = None
        self.labels_info = None
        self.labels_set = None
        self.best_training_score = -np.inf if self.METRIC_FOR_SCORING[self.type]['best_score_is'] == 'higher' else np.inf
        self.are_params_set = False
    

    # public methods

    def get_pipeline(self, use_best_params_if_set=True):
        """
        Instantiates and returns a Scikit-Learn Pipeline object created from this model's individual steps.

        Keyword arguments: 
        use_best_params_if_set -- True if the pipeline should be initialized with its optimal parameters (if they are set),
                                  and False if the default parameters should be used (default True).
        
        Return: 
        Scikit-Learn Pipeline object created from this model's individual steps and optimal parameters (if set).
        """
        # instantiate the Pipeline object
        pipeline_steps = [(name, step()) for name, step in zip(self.steps_name, self.steps)]
        pipeline = Pipeline(steps=pipeline_steps)
        
        if self.bagging_strategy is not None:
            n_estimators = 10
            pipeline = self.bagging_strategy(pipeline, n_jobs=-1, n_estimators=n_estimators, max_samples=1.0 / n_estimators)
        if self.multiclass_strategy is not None:
            pipeline = self.multiclass_strategy(pipeline)
            
        if self.are_params_set and use_best_params_if_set:
            pipeline.set_params(**self.best_params)
        elif self.default_params is not None:
            pipeline.set_params(**self.default_params)

        return pipeline

    def set_params(self, gs_best_params):
        """
        Sets the model's optimal parameters.

        Keyword arguments: 
        gs_best_params -- Dict with keys being parameters' names and values being the optimal parameter value.
        
        Return: -
        """
        self.best_params = gs_best_params
        self.are_params_set = True

    def get_params_ranges(self):
        """
        Returns the model parameters' ranges.

        Keyword arguments: -
        
        Return:
        Dict with keys being parameters' names and values being ranges of possible parameter value.
        """
        updated_params_ranges = {}
        for param_key in self.params_ranges.keys():
            new_param_key = param_key

            if self.bagging_strategy is not None:
                new_param_key = 'base_estimator__' + new_param_key
                
            if self.multiclass_strategy is not None:
                new_param_key = 'estimator__' + new_param_key
               
            updated_params_ranges[new_param_key] = self.params_ranges[param_key]
        return updated_params_ranges

    def get_nb_gridsearch_iter(self, gs_iter_range):
        """
        Returns the number of gridsearch iterations to perform for this model.

        Keyword arguments: 
        gs_iter_range -- dict specifying the number of gridsearch iterations to perform depending on the 
                         model's training speed (training efficiency).
        
        Return:
        Number of gridsearch iterations to perform for this model.
        """
        if self.training_speed_factor == 1:
            return gs_iter_range['fast']
        elif self.training_speed_factor == 2:
            return gs_iter_range['medium']
        elif self.training_speed_factor == 3:
            return gs_iter_range['slow']
        else:
            raise Exception('Invalid "training_speed_factor" parameter:', self.training_speed_factor)

    def train_and_eval(self, X_train, y_train, X_val, y_val, features_name, labels_info, labels_set, plot_cm=False):
        """
        Trains and evaluates the model on the given train and validation data. 
        
        Keyword arguments: 
        X_train -- numpy array of train entries
        y_train -- numpy array of train entries' labels
        X_val -- numpy array of validation entries
        y_val -- numpy array of validation entries' labels
        features_name -- training set's features name
        labels_info -- dict specifying the labels' properties
        labels_set -- list of unique labels that may appear in y_train and y_val
        plot_cm -- True if a confusion matrix plot should be created at the end of evaluation, false otherwise (default: False)
        
        Return: 
        1. dict of scores measured during the pipeline's evaluation
        2. tuple containing a Matplotlib figure and the confusion matrix' values if plot_cm is True None otherwise
        """
        self.features_name = features_name
        self.labels_info = labels_info
        self.labels_set = labels_set

        # train model
        start_time = time.time()
        trained_pipeline = self.get_pipeline(use_best_params_if_set=self.are_params_set).fit(X_train, y_train)
        end_time = time.time()
        
        # evaluate the model
        model, y_pred = self.predict(X_val, compute_proba=self.labels_info['type']=='monolabels', pipeline=trained_pipeline)
        scores, cm = self.eval(y_val, y_pred, model.classes_, plot_cm=plot_cm)

        # measure the best training score & save the trained_pipeline if this new score is the best
        current_score = scores[self.METRIC_FOR_SCORING[self.type]['metric']]
        if self.METRIC_FOR_SCORING[self.type]['best_score_is'] == 'higher':
            if self.best_training_score < current_score:
                self.best_training_score = current_score
                self.best_cv_trained_pipeline = trained_pipeline
        elif self.METRIC_FOR_SCORING[self.type]['best_score_is'] == 'lower':
            if self.best_training_score > current_score:
                self.best_training_score = current_score
                self.best_cv_trained_pipeline = trained_pipeline
        else: 
            raise Exception('Invalid value for variable: RecommendationModel.METRIC_FOR_SCORING["..."]["best_score_is"]: ' + 
                            self.METRIC_FOR_SCORING[self.type]['best_score_is'])

        return scores, cm
    
    def predict(self, X, compute_proba=False, pipeline=None, use_pipeline_prod=True):
        """
        Uses a trained pipeline to predict the labels of the given time series.
        
        Keyword arguments: 
        X -- numpy array of entries to label
        compute_proba -- True if the probability of the sample for each class in the model should be returned, False
                         if only the label should be returned (default: False)
        pipeline -- trained pipeline to use to get predictions. If None, uses either the "production" pipeline or the
                    best pipeline from the cross-validation training (default: None). 
        use_pipeline_prod -- True to use the production pipeline trained on all data, False to use the last pipeline
                             trained during cross-validation (default: True)
        
        Return: 
        1. Model which was used to get recommendations
        2. Numpy array of labels
        """
        if pipeline is None and use_pipeline_prod and self.trained_pipeline_prod is None:
            raise Exception('This model has not been trained on all data after its evaluation. The argument "use_prod_model" '\
                            + 'cannot be set to True.')

        if pipeline is not None:
            trained_pipeline = pipeline
        elif use_pipeline_prod:
            trained_pipeline = self.trained_pipeline_prod
        else:
            trained_pipeline = self.best_cv_trained_pipeline

        if self.type == 'classifier' and compute_proba:
            y_pred = self._custom_predict_proba(trained_pipeline)(X) # probability vector
        else:
            y_pred = trained_pipeline.predict(X)

        return trained_pipeline, y_pred

    def get_recommendations(self, X, use_pipeline_prod=True):
        """
        Uses a trained pipeline to get recommendations for the given time series.
        
        Keyword arguments: 
        X -- numpy array of entries to get recommendations for
        use_pipeline_prod -- True to use the production pipeline trained on all data, False to use the last pipeline
                             trained during cross-validation (default: True)
        
        Return: 
        Numpy array of recommendations
        """
        model, y_pred = self.predict(X, compute_proba=True, use_pipeline_prod=use_pipeline_prod)

        if self.type == 'classifier':
            y_pred_proba = y_pred
            # predict proba vector to dataframe
            preds = pd.DataFrame(y_pred_proba, 
                                 columns=model.classes_)
            preds.index.name = 'Time Series ID'
            return preds

    def eval(self, y_true, y_pred, classes, plot_cm=False):
        """
        Evaluates the model on the given validation/test data. 
        
        Keyword arguments: 
        y_true -- numpy array of true labels
        y_pred -- numpy array of validation/test entries' labels
        plot_cm -- True if a confusion matrix plot should be created at the end of evaluation, false otherwise (default: False)
        
        Return: 
        1. dict of scores measured during the pipeline's evaluation
        2. tuple containing a Matplotlib figure and the confusion matrix' values if plot_cm is True None otherwise
        """
        if self.type == 'classifier':
            are_multi_labels = self.labels_info['type'] == 'multilabels'

            if not are_multi_labels:
                y_pred_proba = y_pred
                y_pred = [classes[np.argmax(probas)] for probas in y_pred_proba]

            average_strat = 'samples' if are_multi_labels else 'weighted'

            scores = {
                'Accuracy': accuracy_score(y_true, y_pred, normalize=True, sample_weight=None), 
                'F1-Score': f1_score(y_true=y_true, y_pred=y_pred, average=average_strat, zero_division=0),
                'Precision': precision_score(y_true=y_true, y_pred=y_pred, average=average_strat, zero_division=0).tolist(), 
                'Recall': recall_score(y_true=y_true, y_pred=y_pred, average=average_strat, zero_division=0).tolist(),
                'Hamming Loss': hamming_loss(y_true, y_pred),
            }
            
            if not are_multi_labels: # compute MRR only for mono labels classifiers
                ranks = [list({b:a for a,b in sorted(zip(probas, classes), reverse=True)}.keys()).index(y_true_i) + 1 
                         for y_true_i, probas in zip(y_true, y_pred_proba)] # rank at which each correct label is found

                scores['Mean Reciprocal Rank'] = (1 / len(y_true)) * sum(1 / rank_i for rank_i in ranks)
                scores['Precision@3'] = sum(int(rank_i <= 3) for rank_i in ranks) / (len(y_true) * 3)
                scores['Recall@3'] = sum(int(rank_i <= 3) for rank_i in ranks) / len(y_true)

            if plot_cm:
                fig, _, cm_val = Utils.plot_confusion_matrix(y_true, y_pred, are_multi_labels, 
                                                            normalize=True, labels=self.labels_set, 
                                                            verbose=0)

        elif self.type == 'regression':
            raise Exception('Regression models are not supported yet.')

        return scores, (fig, cm_val) if plot_cm else None

    def save(self, results_dir):
        """
        Saves a RecommendationModel instance to disk (serialization).

        Keyword arguments: 
        results_dir -- path to the results' directory that will or already does store those files

        Return:
        1. Filename of a serialized RecommendationModel instance
        2. Filename of a serialized trained_pipeline
        """
        model_filename, model_tp_filename, model_tpp_filename = RecommendationModel._get_filenames(self.name, results_dir)

        # serialize the RecommendationModel instance but not its trained_pipeline (using pickle)
        with open(model_filename, 'wb') as model_f_out:
            p_dump(self, model_f_out)

        # serialize the model's best_cv_trained_pipeline (using joblib)
        with open(model_tp_filename, 'wb') as model_f_out:
            j_dump(self.best_cv_trained_pipeline, model_f_out)

        # serialize the model's trained_pipeline_prod (using joblib)
        if self.trained_pipeline_prod is not None:
            with open(model_tpp_filename, 'wb') as model_f_out:
                j_dump(self.trained_pipeline_prod, model_f_out)

        return model_filename, model_tp_filename, model_tpp_filename

    def __repr__(self):
        return '%s: %s' % (self.name, ', '.join([step.__name__ for step in self.steps]))


    # private methods
    
    def __getstate__(self):
        return {k: v for (k, v) in self.__dict__.items() if k not in RecommendationModel._PROPS_TO_NOT_PICKLE}

    def _custom_predict_proba(self, _trained_pipeline):
        """
        Creates a custom predict proba function to support Scikit-Learn classifiers that do not implement a 
        predict_proba function.

        Keyword arguments: 
        _trained_pipeline -- trained classifier pipeline to use to get predictions

        Return:
        Custom predict_proba function that returns the probability of the sample for each class in the model.
        """
        func = None
        try:
            func = _trained_pipeline.predict_proba # the classifier does implement a predict_proba method
        except Exception as e: # the classifier does NOT implement a predict_proba method
            # implement a custom predict_proba method
            def _custom_predict_proba_internal(x):
                prob_pos = _trained_pipeline.decision_function(x)
                prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
                return prob_pos
            func = _custom_predict_proba_internal
        return func


    # static methods
    
    def init_from_descriptions(models_descriptions_to_use):
        """
        Initializes objects of type RecommendationModel from the given list of models' descriptions files.

        Keyword arguments:
        models_descriptions_to_use -- list of models' descriptions files

        Return:
        List of initialized RecommendationModel instances.
        """
        models = []

        # for each model description
        for model_description_fname in models_descriptions_to_use:

            # load description
            spec = importlib.util.spec_from_file_location(
                'Training.ModelsDescription', 
                normp(RecommendationModel.MODELS_DESCRIPTION_DIR + '/' + model_description_fname)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # init model
            model = RecommendationModel(
                name = module.model_info['name'], 
                type = module.model_info['type'], 
                steps = module.model_info['steps'], 
                params_ranges = module.model_info['params_ranges'], 
                training_speed_factor = module.model_info['training_speed_factor'], 

                multiclass_strategy = module.model_info['multiclass_strategy'], 
                bagging_strategy = module.model_info['bagging_strategy'],
                description_filename = model_description_fname,
                default_params = module.model_info['default_params'],
            )
            
            models.append(model)
        return models

    def load_from_archive(archive, repr):
        """
        Loads and returns a RecommendationModel instance.

        Keyword arguments: 
        archive -- archive object containing the RecommendationModel serialized files to load
        repr -- string identifying a model

        Return:
        RecommendationModel instance
        """
        model_filename, model_tp_filename, model_tpp_filename = RecommendationModel._get_filenames(repr, '')

        # load RecommendationModel instance (using pickle)
        with archive.open(os.path.split(model_filename)[-1], 'r') as model_file:
            model = p_load(model_file)

        # load its best_cv_trained_pipeline (using joblib)
        with archive.open(os.path.split(model_tp_filename)[-1], 'r') as model_tp_file:
            model_tp = j_load(model_tp_file)
            model.best_cv_trained_pipeline = model_tp
            assert model.best_cv_trained_pipeline is not None

        # load its trained_pipeline_prod (using joblib)
        try:
            with archive.open(os.path.split(model_tpp_filename)[-1], 'r') as model_tpp_file:
                model_tpp = j_load(model_tpp_file)
            model.trained_pipeline_prod = model_tpp
        except KeyError: # it is possible, the model has not been trained on all data to prepare for production
            model.trained_pipeline_prod = None

        return model
    
    def _get_filenames(repr, results_dir):
        """
        Returns the filenames of a RecommendationModel instance from its repr.

        Keyword arguments: 
        repr -- string identifying a model
        results_dir -- path to the results' directory that will or already does store those files

        Return: 
        1. Filename of a serialized RecommendationModel instance
        2. Filename of a serialized trained_pipeline
        2. Filename of a serialized trained_pipeline_prod
        """
        return (
            normp(results_dir + f'/{repr}.p'), # Recommendation instance filename
            normp(results_dir + f'/{repr}_pipe.joblib'), # trained_pipeline filename
            normp(results_dir + f'/{repr}_pipe_prod.joblib'), # trained_pipeline_prod filename
        )