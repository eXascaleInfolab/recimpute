"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
ModelsTrainer.py
@author: @chacungu
"""

import itertools
import operator
import pandas as pd
import numpy as np
import random as rdm
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split as sklearn_train_test_split
from tqdm.notebook import tqdm

from Training.RecommendationModel import RecommendationModel
from Training.TrainResults import TrainResults
from Utils.Utils import Utils

class ModelsTrainer:
    """
    Class which handles classification / regression models and provides methods for its training and evaluation.
    """

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

    def train(self, train_on_all_data=False):
        """
        Trains and evaluates models given to this initialization of this trainer. Uses T-Daub for best-algorithms selection
        and cross-validation.

        Keyword arguments: 
        train_on_all_data -- True if the models should be trained on ALL data once the cross-val is done, False 
                             otherwise (default: False)

        Return:
        A TrainResults' instance containing the training results
        """
        t_daub_nb_runs = ModelsTrainer.CONF['TDAUB_NB_RUNS']
        nb_best_models = ModelsTrainer.CONF['TDAUB_NB_BEST_MODELS']
        models_to_train = self._t_daub(t_daub_nb_runs, nb_best_models) if t_daub_nb_runs > 0 else self.models
        
        train_results = self._train(models_to_train, train_on_all_data=train_on_all_data, gridsearch=self.CONF['USE_GRIDSEARCH'])
        return train_results
        
    
    # private methods

    def _select(self, recommendation_models, compute_score, training_set_params=None, S=[3, 8, 20, 35, 55, 80], test_method=ttest_rel, p_value=.01):
        """
        Trains and evaluates a list of models over cross-validation (and after gridsearch if it is necessary).

        Keyword arguments:
        recommendation_models -- list of RecommendationModel instances that should be trained and evaluated
        compute_score -- method that computes the score based on which the models are evaluated. Takes as input the F1-score, Recall@3, and
                         training runtime for a given model. Returns a floating point value.
        training_set_params -- dict specifying the data's properties (e.g. should it be balanced, reduced, etc.) (default: None)
        S -- list of percentages for the partial training data sets (default: [3, 8, 20, 35, 55, 80])
        test_method -- Significance test. Takes as input two lists of equal length and returns a tuples which 2nd value is the 
                       p-value. (default: scipy.stats.ttest_rel)
        p_value -- p_value used with the paired t-test to decide if a difference is significant or not (default: 0.01).

        Return:
        List of selected RecommendationModel.
        """
        training_set_params = self.training_set.get_default_properties() if training_set_params is None else training_set_params

        all_data, _, labels_set, X_train, y_train, X_val, y_val = next(self.training_set.yield_splitted_train_val(training_set_params, 1))
        X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
        assert X_train.index.identical(y_train.index)
        print('0/3 - data loaded') # TODO tmp print

        class TmpPipeline:
            def __init__(self, rm):
                self.rm = rm
                self.scores = []

        pipelines = [TmpPipeline(rm) for rm in recommendation_models]

            
        
        train_index = []

        for i in tqdm(range(len(S))):
            n = X_train.shape[0] // (S[i] - (S[i-1] if i > 0 else 0)) # number of data to add for partial training

            #new_pipes = None # TODO sample new pipelines from the ones in "pipelines" (=crossovers btw same clfs)
            #pipelines.extend(new_pipes)
            
            # prepare the partial training set
            X_train_unused = X_train.loc[~X_train.index.isin(train_index)]
            y_train_unused = y_train.loc[~y_train.index.isin(train_index)]
            assert X_train_unused.index.identical(y_train_unused.index)

            try:
                train_index_new = sklearn_train_test_split(X_train_unused, stratify=y_train_unused, train_size=n)[0].index.tolist()
            except:
                train_index_new = sklearn_train_test_split(X_train_unused, train_size=n)[0].index.tolist()
            assert all(id not in train_index for id in train_index_new)

            train_index.extend(train_index_new)
            print('1/3 - begining of new partial training: %i%% -> %i' % (S[i], len(train_index))) # TODO tmp print
            
            Xp_train = X_train.loc[train_index]
            yp_train = y_train.loc[train_index]
            assert Xp_train.index.identical(yp_train.index)

            # train, eval and perform statistic tests
            metrics = {}
            for pipe in tqdm(pipelines, leave=False):
                metrics[pipe.rm.id] = []
                n_splits = 10 if i < 1 else 2
                skf = StratifiedKFold(n_splits=n_splits)

                # cross-validation
                for train_index_cv, _ in tqdm(skf.split(Xp_train, yp_train), total=n_splits, leave=False):
                    #print('2/3 - cv split') # TODO tmp print
                    # skf returns row indices (0-nb_rows) NOT pandas dataframe's index !!! thus the use of iloc[] and not loc[]
                    Xp_train_cv = Xp_train.iloc[train_index_cv]
                    yp_train_cv = yp_train.iloc[train_index_cv]
                    assert Xp_train_cv.index.identical(yp_train_cv.index)
                    Xp_train_cv = Xp_train_cv.to_numpy().astype('float32')
                    yp_train_cv = yp_train_cv.to_numpy().astype('str').flatten()

                    # training
                    with Utils.catchtime('Training pipe %s' % pipe.rm.pipe, verbose=False) as t:
                        metrics_, _ = pipe.rm.train_and_eval(Xp_train_cv, yp_train_cv, X_val, y_val, 
                                                             all_data.columns, self.training_set.get_labeler_properties(), labels_set,
                                                             plot_cm=False, save_if_best=False)
                    runtime = t.end - t.start
                    metrics[pipe.rm.id].append((metrics_['F1-Score'], metrics_['Recall@3'], runtime))

            print('3/3 - statistic tests') # TODO tmp print
            # normalize runtime between 0 and 1 & compute the score of each pipeline on each cv-split
            max_runtime = max([i[2] for cv_scores in metrics.values() for i in cv_scores])
            scores_ = {id: [compute_score(f1,r3,t/max_runtime) for f1,r3,t in cv_scores] for id, cv_scores in metrics.items()}
            
            # add the newly measured scores to the global scores list
            for pipe in pipelines:
                pipe.scores.extend(scores_[pipe.rm.id])
            
            # statistical tests - remove pipes that are significantly worse than any other pipe
            worse_pipes = self._apply_test(
                list(itertools.combinations(pipelines, r=2)), 
                test_method,
                p_value=p_value
            )

            print([(p.rm.id, np.mean(p.scores)) for p in pipelines])
            print([
                (test_method(a.scores, b.scores)[1], a.rm.id,b.rm.id)  # we keep the worse pipelines of the 2
                    for a,b in itertools.combinations(pipelines, r=2) # for all pairs of pipes
            ])

            # remove the worse pipes
            pipelines = [p for p in pipelines if p not in worse_pipes]

            print('There remains %i pipelines. %i have been eliminated.' % (len(pipelines), len(worse_pipes)))

            if len(pipelines) < 3:
               break

        return [pipe.rm for pipe in pipelines]

    def _apply_test(self, pipes_combinations, test_method, p_value=.01):
        """
        Applies a significance test (paired t-test) to the pipelines to identify and return those that perform worse than others.

        Keyword arguments:
        pipes_combinations -- list of TmpPipeline to apply t-test to.
        test_method -- Significance test. Takes as input two lists of equal length and returns a tuples which 2nd value is the p-value.
        p_value -- p_value used with the paired t-test to decide if a difference is significant or not (default: 0.01).

        Return:
        List of TmpPipeline that is performing worse than others.
        """
        worse_pipes = []
        i = len(pipes_combinations)-1
        while i > 0:
            a,b = pipes_combinations[i]
            # if the paired t-test shows a statistical difference between the two
            if test_method(a.scores, b.scores)[1] < p_value:
                # eliminate the worst one
                eliminated_pipe = a if np.mean(a.scores) < np.mean(b.scores) else b
                worse_pipes.append(eliminated_pipe)
                
                # remove all pairs of pipes continaining the eliminated pipe to avoid unnecessary comparisons
                for j in reversed(range(len(pipes_combinations))):
                    x,y = pipes_combinations[j]
                    if eliminated_pipe in (x,y):
                        del pipes_combinations[j]
                        i -= 1 if j < i else 0
            i -= 1
        return worse_pipes

    def _train(self, models_to_train, train_on_all_data=False, training_set_params=None, save_results=True, save_if_best=True):
        """
        Trains and evaluates a list of models over cross-validation (and after gridsearch if it is necessary).

        Keyword arguments:
        models_to_train -- list of RecommendationModel instances that should be trained and evaluated
        train_on_all_data -- True if the models should be trained on ALL data once the cross-val is done, False 
                             otherwise (default: False)
        training_set_params -- dict specifying the data's properties (e.g. should it be balanced, reduced, etc.) (default: None)
        save_results -- True if the results should be saved to disk, False otherwise (default: True)
        save_if_best -- True if the model should be saved if it is the best performing one, false otherwise (default: False)

        Return:
        A TrainResults' instance containing the training results
        """
        training_set_params = self.training_set.get_default_properties() if training_set_params is None else training_set_params

        train_results = TrainResults(models_to_train, self.training_set.get_labeler_properties()['type'])
        try:
            for split_id, yielded in enumerate(self.training_set.yield_splitted_train_val(training_set_params, 
                                                                                          ModelsTrainer.CONF['NB_CV_SPLITS'])):
                print('\nCross-validation split n°%i' % (split_id+1))
                all_data, all_labels, labels_set, X_train, y_train, X_val, y_val = yielded

                print('X_train shape:', X_train.shape, ', X_val shape:', X_val.shape) # TODO tmp print
                print(np.asarray(np.unique(y_train, return_counts=True)).T) # TODO tmp print
                print(np.asarray(np.unique(y_val, return_counts=True)).T) # TODO tmp print

                for model in models_to_train:

                    # training
                    print('Training %s.' % model)
                    with Utils.catchtime('Training model %s @ split %i' % (model, split_id)):
                        scores, cm = model.train_and_eval(X_train, y_train, X_val, y_val, 
                                                          all_data.columns, self.training_set.get_labeler_properties(), labels_set,
                                                          plot_cm=True, save_if_best=save_if_best)

                    cm[0].savefig('Training/Results/Plots/%s: %i' % (model, split_id)) # TODO tmp print

                    # save results
                    # TrainResults contain a dict grouping the results of each trained model
                    train_results.add_model_cv_split_result(split_id, model, scores, cm)

            if train_on_all_data: # train the models on all data: 
                print('\nTraining models on all data.')
                # Warning this model should not be used for any kind of evaluation but only for production use
                _, X_all, y_all = self.training_set.get_all_data(training_set_params)
                for model in models_to_train:
                    with Utils.catchtime('Training model %s on all data' % model):
                        model.trained_pipeline_prod = clone(model.pipe).fit(X_all, y_all)

        finally:
            # save results to disk
            if save_results:
                train_results.save(self.training_set, save_train_set=ModelsTrainer.CONF['SAVE_TRAIN_SET'])

        return train_results

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
            train_results = self._train(self.models, training_set_properties, save_results=False)
            
            # retrieve the F1-Score of each model from the results
            for model in train_results.models:
                model_results = train_results.get_avg_model_results(model)
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
        best_models = [elem[0] for elem in sorted_ranking[:nb_best_models]]

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