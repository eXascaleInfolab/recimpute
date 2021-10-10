"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
RecommendationModel.py
@author: @chacungu
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, f1_score
import time

from Utils.Utils import Utils

class RecommendationModel:
    """
    Class which handles a recommendation model (classification / regression model).
    """

    #


    # constructor
    def __init__(self, name, pipeline):
        self.pipeline = pipeline
        assert hasattr(self.pipeline[-1], 'fit') and ismethod(getattr(self.pipeline[-1], 'fit'))
        self.params = None # TODO
    

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