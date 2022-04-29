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
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, f1_score
import time

from Utils.Utils import Utils

class RecommendationModel:
    """
    Class which handles a recommendation model (classification / regression model).
    """

    MODELS_DESCRIPTION_DIR = normp('./Training/ModelsDescription/')
    METRICS_CLF_MONO = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Mean Reciprocal Rank', 'Precision@3', 'Recall@3']
    METRICS_CLF_MULTI = ['Accuracy', 'F1-Score', 'Precision', 'Recall'] # , 'Hamming Loss'
    METRICS_REGR = ['Mean Squared Error']
    METRIC_FOR_SCORING = {
        'classifier': {'metric': 'F1-Score', 'best_score_is': 'higher'},
        'regression': {'metric': 'Mean Squared Error', 'best_score_is': 'lower'},
    }
    _PROPS_TO_NOT_PICKLE = ['best_cv_trained_pipeline', 'trained_pipeline_prod'] # these are handled separately


    # constructor
    def __init__(self, id, type, pipe):
        """
        Initializes a RecommendationModel object.

        Keyword arguments:
        id -- unique id of this model
        type -- type of this model (classifier or regression model)
        pipe -- sklearn pipeline object with each step already set up with their parameters
        """
        self.id = id
        self.type = type
        self.pipe = pipe
        assert type in ['classifier', 'regression']

        self.best_cv_trained_pipeline = None
        self.trained_pipeline_prod = None
        self.features_name = None
        self.labels_info = None
        self.labels_set = None
        self.best_training_score = -np.inf if self.METRIC_FOR_SCORING[self.type]['best_score_is'] == 'higher' else np.inf
        self.are_params_set = False
    

    # public methods

    def train_and_eval(self, X_train, y_train, X_val, y_val, features_name, labels_info, labels_set, plot_cm=False, save_if_best=True):
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
        save_if_best -- True if the model should be saved if it is the best performing one, false otherwise (default: False)
        
        Return: 
        1. dict of scores measured during the pipeline's evaluation
        2. tuple containing a Matplotlib figure and the confusion matrix' values if plot_cm is True None otherwise
        """
        self.features_name = features_name
        self.labels_info = labels_info
        self.labels_set = labels_set

        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        # train model
        start_time = time.time()
        trained_pipeline = clone(self.pipe).fit(X_train, y_train)
        end_time = time.time()
        
        # evaluate the model
        model, y_pred = self.predict(X_val, compute_proba=self.labels_info['type']=='monolabels', pipeline=trained_pipeline)
        scores, cm = self.eval(y_val, y_pred, model.classes_, plot_cm=plot_cm)

        # measure the best training score & save the trained_pipeline if this new score is the best
        current_score = scores[self.METRIC_FOR_SCORING[self.type]['metric']]
        if save_if_best:
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
        trained_pipeline = self.get_trained_pipeline(pipeline=pipeline, use_pipeline_prod=use_pipeline_prod)

        if self.type == 'classifier' and compute_proba:
            y_pred = self._custom_predict_proba(trained_pipeline)(X) # probability vector
        else:
            y_pred = trained_pipeline.predict(X)

        return trained_pipeline, y_pred

    def get_trained_pipeline(self, pipeline=None, use_pipeline_prod=True):
        """
        Returns the appropriate trained pipeline to use to get predictions.

        Keyword arguments:
        pipeline -- trained pipeline to use to get predictions. If None, uses either the "production" pipeline or the
                    best pipeline from the cross-validation training (default: None). 
        use_pipeline_prod -- True to use the production pipeline trained on all data, False to use the last pipeline
                             trained during cross-validation (default: True)

        Return:
        Trained pipeline to use to get predictions.
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
        return trained_pipeline

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

    def eval(self, y_true, y_pred, classes, categories=None, plot_cm=False):
        """
        Evaluates the model on the given validation/test data. 
        
        Keyword arguments: 
        y_true -- numpy array of true labels
        y_pred -- numpy array of validation/test entries' labels
        classes -- list of classes
        categories -- numpy array in the same order as y_true and y_pred that contains the category of the dataset of each sequence. If not None, 
                      the evaluation is done per category (default. None)
        plot_cm -- True if a confusion matrix plot should be created at the end of evaluation, false otherwise (default: False)
        
        Return: 
        1. dict of scores measured during the pipeline's evaluation
        2. tuple containing a Matplotlib figure and the confusion matrix' values if plot_cm is True None otherwise
        3. if categories is not None: dict with keys being the categories and values their dict of scores and confusion matrix
           otherwise None
        """
        
        global_results = self._eval(y_true, y_pred, classes, plot_cm=plot_cm)
        if categories is not None:
            results_per_categories = {}
            for category in np.unique(categories):
                ix = np.where(categories == category)
                if len(ix) > 0:
                    res = self._eval(y_true[ix], y_pred[ix], classes, plot_cm=plot_cm)
                    results_per_categories[category] = res
            return *global_results, results_per_categories
        return global_results

    def save(self, results_dir):
        """
        Saves a RecommendationModel instance to disk (serialization).

        Keyword arguments: 
        results_dir -- path to the results' directory that will or already does store those files

        Return:
        1. Filename of a serialized RecommendationModel instance
        2. Filename of a serialized trained_pipeline
        """
        model_filename, model_tp_filename, model_tpp_filename = RecommendationModel._get_filenames(self.id, results_dir)

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
        return '%s: %s' % (self.id, ', '.join([str(step) for (_, step) in self.pipe.steps]))


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

    def _eval(self, y_true, y_pred, classes, plot_cm=False):
        """
        Evaluates the model on the given validation/test data. 
        
        Keyword arguments: 
        y_true -- numpy array of true labels
        y_pred -- numpy array of validation/test entries' labels
        classes -- list of classes
        plot_cm -- True if a confusion matrix plot should be created at the end of evaluation, false otherwise (default: False)
        
        Return: 
        1. dict of scores measured during the pipeline's evaluation
        2. tuple containing a Matplotlib figure and the confusion matrix' values if plot_cm is True None otherwise
        3. if categories is not None: dict with keys being the categories and values their dict of scores and confusion matrix
           otherwise None
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
                #'Hamming Loss': hamming_loss(y_true, y_pred),
            }
            
            if not are_multi_labels: # todo those metrics should also be computed for multi-labels classifiers
                get_sorted_recommendations = lambda probas, classes: list({b:a for a,b in sorted(zip(probas, classes), reverse=True)}.keys())
                # rank at which each y_true is found in the sorted y_pred_proba (list of len = len(y_true))
                ranks = [get_sorted_recommendations(y_pred_proba_i, classes).index(y_true_i) + 1 
                        if y_true_i in classes else
                        np.inf
                        for y_true_i, y_pred_proba_i in zip(y_true, y_pred_proba)] 

                scores['Mean Reciprocal Rank'] = (1 / len(y_true)) * sum(1 / rank_i for rank_i in ranks)
                # average prec@K and recall@k
                K = 3
                prec_at_k = lambda K, rank_y_true: int(rank_y_true <= K) / K
                scores['Precision@3'] = sum(prec_at_k(K, rank_y_true) for rank_y_true in ranks) / len(y_true)
                recall_at_k = lambda K, rank_y_true: int(rank_y_true <= K) / 1
                scores['Recall@3'] = sum(recall_at_k(K, rank_y_true) for rank_y_true in ranks) / len(y_true)

            if plot_cm:
                fig, _, cm_val = Utils.plot_confusion_matrix(y_true, y_pred, are_multi_labels, 
                                                            normalize=True, labels=self.labels_set, 
                                                            verbose=0)

        elif self.type == 'regression':
            raise Exception('Regression models are not supported yet.')

        return scores, (fig, cm_val) if plot_cm else None

    # static methods
    
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