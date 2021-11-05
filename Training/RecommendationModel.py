"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
RecommendationModel.py
@author: @chacungu
"""

import importlib.util
from joblib import dump as j_dump, load as j_load
import os
from os.path import normpath as normp
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
    _PROPS_TO_NOT_PICKLE = ['trained_pipeline', 'trained_pipeline_prod']


    # constructor
    def __init__(self, name, steps, params_ranges, training_speed_factor, 
                 multiclass_strategy=None, bagging_strategy=None, description_filename=None):
        """
        Initializes a RecommendationModel object.

        Keyword arguments:
        name -- name of this model
        steps -- list of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they 
                 are chained, with the last object an estimator
        params_ranges -- dict with keys being parameters' names and values being ranges of possible parameter value
        training_speed_factor -- integer between 1 (fast) and 3 (slow) used to indicate the expected training speed of this model
        multiclass_strategy -- Class of a multiclass strategy (default: None)
        bagging_strategy -- Class of a bagging strategy (default: None)
        description_filename -- filename of this model's description file (default: None)
        """
        self.name = name
        self.steps_name, self.steps = zip(*steps)
        # for step in self.steps:
        #     assert hasattr(step, 'fit') and callable(getattr(step, 'fit'))
        #     assert hasattr(step, 'predict') and callable(getattr(step, 'predict'))
        #     assert hasattr(step, 'transform') and callable(getattr(step, 'transform'))

        self.params_ranges = params_ranges
        self.best_params = None
        self.training_speed_factor = training_speed_factor
        self.multiclass_strategy = multiclass_strategy
        self.bagging_strategy = bagging_strategy
        self.description_filename = description_filename

        self.trained_pipeline = None # TODO do we still need to store this since we god the trained_pipeline_prod ?
        self.trained_pipeline_prod = None
        self.features_name = None
        self.labels_info = None
        self.labels_set = None
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
        self.trained_pipeline = self.get_pipeline().fit(X_train, y_train)
        end_time = time.time()
        
        # evaluate the model
        scores, cm = self.eval(X_val, y_val, plot_cm)

        return scores, cm
    
    def eval(self, X, y, plot_cm=False):
        """
        Evaluates the model on the given validation/test data. 
        
        Keyword arguments: 
        X -- numpy array of validation/test entries
        y -- numpy array of validation/test entries' labels
        plot_cm -- True if a confusion matrix plot should be created at the end of evaluation, false otherwise (default: False)
        
        Return: 
        1. dict of scores measured during the pipeline's evaluation
        2. tuple containing a Matplotlib figure and the confusion matrix' values if plot_cm is True None otherwise
        """
        # predict new data's labels
        y_pred = self.trained_pipeline.predict(X)

        are_multi_labels = self.labels_info['type'] == 'multilabels'
        average_strat = 'samples' if are_multi_labels else 'weighted'

        scores = {
            'Accuracy': accuracy_score(y, y_pred, normalize=True, sample_weight=None), 
            'F1-Score': f1_score(y_true=y, y_pred=y_pred, average=average_strat, zero_division=0),
            'Precision': precision_score(y_true=y, y_pred=y_pred, average=average_strat, zero_division=0).tolist(), 
            'Recall': recall_score(y_true=y, y_pred=y_pred, average=average_strat, zero_division=0).tolist(),
            'Hamming Loss': hamming_loss(y, y_pred),
        }

        if plot_cm:
            fig, _, cm_val = Utils.plot_confusion_matrix(y, y_pred, are_multi_labels, 
                                                         normalize=True, labels=self.labels_set, 
                                                         verbose=0)

        return scores, (fig, cm_val) if plot_cm else None

    def predict(self, X, use_pipeline_prod=True):
        """
        Uses a trained pipeline to predict the labels of the given time series.
        
        Keyword arguments: 
        X -- numpy array of entries to label
        use_pipeline_prod -- True to use the production pipeline trained on all data, False to use the last pipeline
                             trained during cross-validation (default: True)
        
        Return: 
        Numpy array of recommendations
        """
        if use_pipeline_prod and self.trained_pipeline_prod is None:
            raise Exception('This model has not been trained on all data after its evaluation. The argument "use_prod_model" '\
                            + 'cannot be set to True.')
        trained_pipeline = self.trained_pipeline_prod if use_pipeline_prod else self.trained_pipeline
        y = trained_pipeline.predict(X) # TODO implement a more complex recommendation method (top 3?)
        return y

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

        # serialize the model's trained_pipeline (using joblib)
        with open(model_tp_filename, 'wb') as model_f_out:
            j_dump(self.trained_pipeline, model_f_out)

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
                module.model_info['name'], 
                module.model_info['steps'], 
                module.model_info['params_ranges'], 
                module.model_info['training_speed_factor'], 
                module.model_info['multiclass_strategy'], 
                module.model_info['bagging_strategy'],
                model_description_fname,
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

        # load its trained_pipeline (using joblib)
        with archive.open(os.path.split(model_tp_filename)[-1], 'r') as model_tp_file:
            model_tp = j_load(model_tp_file)
        model.trained_pipeline = model_tp

        # load its trained_pipeline (using joblib)
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