"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
RecommendationModel.py
@author: @chacungu
"""

import importlib.util
from os.path import normpath as normp
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, f1_score
import time

from Utils.Utils import Utils

class RecommendationModel:
    """
    Class which handles a recommendation model (classification / regression model).
    """

    MODELS_DESCRIPTION_DIR = normp('./Training/ModelsDescription/')


    # constructor
    def __init__(self, name, steps, params_range, multiclass_strategy=None, bagging_strategy=None):
        self.steps = steps
        assert hasattr(self.steps[-1], 'fit') and callable(getattr(self.steps[-1], 'fit'))

        self.params_range = params_range
        self.multiclass_strategy = multiclass_strategy
        self.bagging_strategy = bagging_strategy
    

    # public methods

    def are_params_set(self):
        # TODO
        pass

    def get_params_ranges(self):
        # TODO
        pass

    def get_nb_gridsearch_iter(self):
        # TODO
        pass

    def get_pipeline(self):
        # TODO
        pass

    def train_and_eval(self, X_train, y_train, X_val, y_val, labels_info, labels_set, plot_cm=False):
        """
        Trains and evaluates the model on the given train and validation data. 
        
        Keyword arguments: 
        X_train -- numpy array of train entries
        y_train -- numpy array of train entries' labels
        X_val -- numpy array of validation entries
        y_val -- numpy array of validation entries' labels
        labels_info -- dict specifying the labels' properties
        labels_set -- list of unique labels that may appear in y_train and y_val
        plot_cm -- True if a confusion matrix plot should be created at the end of evaluation, false otherwise (default: False)
        
        Return: 
        1. trained pipeline
        2. list of scores measured during the pipeline's evaluation
        3. tuple containing a Matplotlib figure and the confusion matrix' values if plot_cm is True None otherwise
        """
        # train model
        start_time = time.time()
        trained_pipeline = self.get_pipeline().fit(X_train, y_train) # TODO
        end_time = time.time()

        # predict new data's labels
        y_pred = trained_pipeline.predict(X_val)

        are_multi_labels = labels_info['type'] == 'multilabels'
        average_strat = 'samples' if are_multi_labels else 'weighted'

        scores = [
            accuracy_score(y_val, y_pred, normalize=True, sample_weight=None), 
            precision_score(y_true=y_val, y_pred=y_pred, average=average_strat, zero_division=0).tolist(), 
            recall_score(y_true=y_val, y_pred=y_pred, average=average_strat, zero_division=0).tolist(),
            hamming_loss(y_val, y_pred),
            f1_score(y_true=y_val, y_pred=y_pred, average=average_strat, zero_division=0),
        ]

        if plot_cm:
            fig, _, cm_val = Utils.plot_confusion_matrix(y_val, y_pred, 
                                                        multilabels=are_multi_labels, normalize=True, 
                                                        labels=labels_set) # TODO

        return trained_pipeline, scores, (fig, cm_val) if plot_cm else None
    

    # private methods

    #def

    
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
                module.model_info['params_range'], 
                module.model_info['multiclass_strategy'], 
                module.model_info['bagging_strategy']
            )
            
            models.append(model)
        return models