"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
ModelsTrainer.py
@author: @chacungu
"""

import math
import numpy as np
from sklearn.linear_model import LinearRegression

from Training.RecommendationModel import RecommendationModel
from Training.TrainResults import TrainResults
from Utils.Utils import Utils

class ModelsTrainer:
    """
    Class which handles classification / regression models and provides methods for its training and evaluation.
    """

    #

    CONF = Utils.read_conf_file('modelstrainer')


    # constructor
    def __init__(self, training_set, models):
        """
        Initializes a ModelsTrainer object.

        Keyword arguments:
        training_set -- TrainingSet instance
        models -- list of RecommendationModels' instances
        """
        self.training_set = training_set
        self.models = models
    

    # public methods

    def train():
        t_daub_nb_runs = ModelsTrainer.CONF['TDAUB_NB_RUNS']
        nb_best_models = ModelsTrainer.CONF['TDAUB_NB_BEST_MODELS']
        models_to_train = self._t_daub(t_daub_nb_runs, nb_best_models) if t_daub_nb_runs > 0 else self.models
        
        train_results = self._train() # TODO
        
    
    # private methods

    def _train(training_set_params=None):
        # TODO _gridsearch_and_training
        training_set_params = self.training_set.get_default_properties() if training_set_params is None else training_set_params
        try:
            for cv_split_id in range(ModelsTrainer.CONF['NB_CV_SPLITS']):
                yielded = self.training_set.yield_splitted_train_val(training_set_params) # TODO
                all_data, all_labels, labels_set, X_train, y_train, X_val, y_val = yielded

                for model in self.models:

                    # gridsearch if necessary
                    if not model.are_params_set(): # TODO
                        gs = RandomizedSearchCV(model.get_pipeline(), # TODO
                                                model.get_params_ranges(), # TODO
                                                cv=ModelsTrainer.CONF['GS_NB_CV_SPLITS'], 
                                                n_iter=model.get_nb_gridsearch_iter(), # TODO
                                                scoring='f1_macro')
                        gs.fit(np.array(data.to_numpy()), np.array(labels.iloc[:].tolist()))
                        model.set_params(gs.best_params_)

                    # training
                    trained_pipeline, scores, cm = model.train_and_eval(X_train, y_train, X_val, y_val, 
                                                                        self.training_set.get_labeler_properties(), labels_set)

                    # save results
                    # TODO TrainResults contain a dict grouping the results of each trained model

        finally:
            # TODO save results to disk
            pass

    def _t_daub(self, nb_runs, nb_best_models):
        """
        Uses the T-Daub strategy (from Shah, Syed Yousaf et al. (2021). "AutoAI-TS: AutoAI for Time Series Forecasting") 
        to predict and return the best performing models when trained on all data without actually training them.

        Keyword arguments:
        nb_runs -- number of T-Daub runs (max 15). The more runs, the more precise the predictions will be at the expense
                   of the computation time.
        nb_best_models -- number of models to return (the predicted top N best-performing ones)
        
        Return:
        List of RecommendationModel instances (the predicted top N best-performing ones)
        """
        if nb_runs > 15:
            raise Exception('Number of T-Daub runs cannot exceed 15 iterations (= 85% of training set used).')
        if nb_runs > 0 and nb_best_models > len(self.models):
            raise Exception('Number of models to train on the whole data set is greater than the number of given models.')

        # evaluate each model on different subsets of the data
        all_data_perc = []
        all_models_results = {}
        for run_id in range(nb_runs):
            # get a subset of the data
            usable_data_perc = self._t_daub_get_usable_data_perc(run_id)
            all_data_perc.append(usable_data_perc)

            training_set_properties = self.training_set.get_default_properties()
            training_set_properties['usable_data_perc'] = usable_data_perc
            # train each model on the reduced data set
            train_results = self._train(training_set_properties) # TODO
            
            # retrieve the F1-Score of each model from the results
            for model in train_results.get_all_models(): # TODO
                model_results = train_results.get_model_results(model.name) # TODO
                f1score = model_results['F1-Score']
                try:
                    all_models_results[model].append(f1score)
                except:
                    all_models_results[model] = [f1score]
        
        all_data_perc = np.array(all_data_perc).reshape(-1, 1)

        # sort the models by predicted accuracy
        ranking = {}
        for model, f1scores in all_models_results.items():
            f1scores = np.array(f1scores).reshape(-1, 1)

            # linear regression
            reg = LinearRegression().fit(all_data_perc, f1scores)

            # ranking of the pipelines based on their predicted scores when using all data
            ranking[model] = reg.predict([[1.0]])[0][0]

        # sort ranking
        sorted_ranking = list(sorted(ranking.items(), key=operator.itemgetter(1), reverse=True))
        # selection of the top ranked pipelines
        best_models = sorted_ranking[:nb_best_models]

        return best_models

    def _t_daub_get_usable_data_perc(self, run_id):
        """
        Returns the percentage of data to use for the T-Daub run's specified ID.
        Non-linear range of values see below:
        [0.05, 0.258, 0.38, 0.466, 0.533, 0.588, 0.634, 0.674, 0.709, 0.741, 0.769, 0.795, 0.819, 0.842, 0.862]

        Keyword arguments:
        run_id -- int, ID of the next T-Daub run
        
        Return:
        Percentage of data to use
        """
        start_range = 0.1
        x = np.arange(start_range, 3, 0.1)[run_id]
        used_percentage = round((math.log(x) - math.log(start_range)) * 0.3 + 0.05, 3)
        return used_percentage