"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
ModelsTrainer.py
@author: @chacungu
"""

import itertools
import logging
from multiprocessing import Pool, Manager
import numpy as np
import operator
import pandas as pd
import random as rdm
from scipy.stats import ttest_rel
from sklearn import pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split as sklearn_train_test_split
from tqdm import tqdm # .notebook

from Training.ClfPipeline import ClfPipeline
from Training.RecommendationModel import RecommendationModel
from Training.TrainResults import TrainResults
from Utils.Utils import Utils

logging.basicConfig(filename='_Logs/msa_new_clfs_errors.log') # TODO tmp delete

def _parallel_training(args):
    """
    TODO
    """
    pipe, train_index_cv, rmvd_pipes, Xp_train, yp_train, X_val, y_val, data_cols, labeler_properties, labels_set = args
    if pipe.rm.id not in rmvd_pipes:
        # get the stratified fold data
        Xp_train_cv = Xp_train.iloc[train_index_cv]
        yp_train_cv = yp_train.iloc[train_index_cv]
        assert Xp_train_cv.index.identical(yp_train_cv.index)
        Xp_train_cv = Xp_train_cv.to_numpy().astype('float32')
        yp_train_cv = yp_train_cv.to_numpy().astype('str').flatten()

        # training & evaluation
        try:
            with Utils.catchtime('Training pipe %s' % pipe.rm.pipe, verbose=False) as t:
                metrics_, _ = pipe.rm.train_and_eval(Xp_train_cv, yp_train_cv, X_val, y_val, 
                                                    data_cols, labeler_properties, labels_set,
                                                    plot_cm=False, save_if_best=False)
            runtime = t.end - t.start
            return pipe.rm.id, metrics_['F1-Score'], metrics_['Recall@3'], runtime
        except:
            logging.exception('Error for pipe %s\n\n\n' % pipe) # TODO tmp delete
            return pipe.rm.id


class ModelsTrainer:
    """
    Class which handles classification / regression models and provides methods for its training and evaluation.
    """

    CONF = Utils.read_conf_file('modelstrainer')


    # constructor

    def __init__(self, training_set):
        """
        Initializes a ModelsTrainer object.

        Keyword arguments:
        training_set -- TrainingSet instance
        """
        self.training_set = training_set
        self.models = None
    

    # public methods

    def select(self, pipelines, all_pipelines_txt, S, selection_len, score_margin,
               training_set_params=None, n_splits=3, test_method=ttest_rel, p_value=.01, alpha=1., beta=1., gamma=1., # TODO give good values to alpha,beta,gamma
               allow_early_eliminations=True, early_break=False):
        """
        Selects (and partially trains if pipelines' training can be paused and resumed) the most-promising pipelines.

        Keyword arguments:
        pipelines -- list of ClfPipeline instances to select from
        all_pipelines_txt -- list of tuples containing the description of each pipeline (steps and params)
        S -- list of percentages for the partial training data sets
        selection_len -- approximate number of pipelines that should remain after the selection process is done.
        score_margin -- Scores acceptance margin. During selection, if a pipeline performs below MAX_SCORE - MARGIN it gets 
                        eliminated before even finishing cross-validation. Scores vary between 0 and 1.
        training_set_params -- dict specifying the data's properties (e.g. should it be balanced, reduced, etc.) (default: None)
        n_splits -- number of k-fold stratified splits (default: 3)
        test_method -- Significance test. Takes as input two lists of equal length and returns a tuples which 2nd value is the 
                       p-value. (default: scipy.stats.ttest_rel)
        p_value -- p_value used with the paired t-test to decide if a difference is significant or not (default: 0.01).
        alpha -- alpha parameter in the score function. Weight of F1-Score. Must be between 0 and 1 (default: TODO).
        beta -- beta parameter in the score function. Weight of Recall@3. Must be between 0 and 1 (default: TODO).
        gamma -- gamma parameter in the score function. Weight of training time. Must be between 0 and 1 (default: TODO).
        allow_early_eliminations -- True if early eliminations are allowed, False if every model should finish their cv-partial-training
                                     even if early results show evidences of bad performance (default True).
        early_break -- True if the process can stop before all the iterations are done IF the target number of pipes has been reached,
                       False if all the iterations should be executed before returning. (default False)

        Return:
        List of selected ClfPipeline.
        """
        print('ModelRace config:', S, alpha, beta, gamma, allow_early_eliminations) # TODO tmp print
        try:
            training_set_params = self.training_set.get_default_properties() if training_set_params is None else training_set_params

            all_data, _, labels_set, X_train, y_train, X_val, y_val = next(self.training_set.yield_splitted_train_val(training_set_params, 1))
            X_train, y_train = pd.DataFrame(X_train), pd.DataFrame(y_train)
            assert X_train.index.identical(y_train.index)
            print('\n0/3 - data loaded') # TODO tmp print

            init_nb_pipes = len(pipelines)
            pruning_factor = (init_nb_pipes / selection_len)**(1/len(S))

            get_max_nb_p_at_i = lambda iter_id: round(init_nb_pipes // (pruning_factor**iter_id))

            manager = Manager()
            
            train_index = []

            with Utils.catchtime('ModelRace runtime', verbose=True): # TODO tmp print
                for i in tqdm(range(len(S)), leave=False):
                    with Utils.catchtime('Selection iteration %i' % (i+1), verbose=True): # TODO tmp print
                        n = (X_train.shape[0] * (S[i] - (S[i-1] if i > 0 else 0))) // 100 # number of data to add for partial training

                        # generate new pipelines from the remaining set of candidates
                        if i > 0:
                            max_nb_pipes_at_iter_i = get_max_nb_p_at_i(i) # init_nb_pipes // (pruning_factor**i)
                            nb_pipes_to_generate = max(max_nb_pipes_at_iter_i - len(pipelines), len(pipelines)//10)
                            new_pipes = ClfPipeline.generate_from_set(pipelines, all_pipelines_txt, nb_pipes_to_generate)
                            pipelines.extend(new_pipes)
                            print('\nTried to generate %i new pipelines.' % nb_pipes_to_generate) # TODO tmp print
                            print('Generated %i new pipelines from the remaining candidates (max nb pipes at iter %i is %i).' % (len(new_pipes), i, max_nb_pipes_at_iter_i))
                        
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
                        print('\n1/3 - begining of new partial training: %i%% -> %i' % (S[i], len(train_index))) # TODO tmp print
                        
                        Xp_train = X_train.loc[train_index]
                        yp_train = y_train.loc[train_index]
                        assert Xp_train.index.identical(yp_train.index)

                        # train, eval and perform statistic tests
                        # create params list
                        rmvd_pipes = manager.dict()
                        param_list = []
                        for pipe in pipelines:
                            n_splits_ = n_splits if len(pipe.scores) > 0 else 10 + (i * n_splits)
                            for train_index_cv, _ in StratifiedKFold(n_splits=n_splits_).split(Xp_train, yp_train):
                                param_list.append(
                                    (pipe, train_index_cv, rmvd_pipes, # dynamic params
                                    Xp_train, yp_train, X_val, y_val, all_data.columns, self.training_set.get_labeler_properties(), labels_set) # constant params
                                )
                        rdm.shuffle(param_list) # shuffle the param list such that the premature elimination is more efficient
                        
                        # run _parallel_training
                        metrics = {}
                        max_score = -np.inf
                        with Pool() as pool:
                            for res in tqdm(pool.imap_unordered(_parallel_training, param_list), total=len(param_list), leave=False):
                                if res is not None:
                                    try:
                                        p_id, f1, r3, t = res
                                    except:
                                        # eliminate this pipe since there most-likely was an exception thrown during its training
                                        #  maybe due to a problematic parameters' combination?
                                        p_id = res
                                        rmvd_pipes[p_id] = True
                                        print('%i has been eliminated probably due to an exception being thrown' % p_id)
                                        continue 

                                    # save the evaluation results
                                    if p_id not in metrics:
                                        metrics[p_id] = []
                                    metrics[p_id].append((f1, r3, t))

                                    # early eliminations
                                    if allow_early_eliminations and len(metrics[p_id]) >= 5:
                                        mean_score = np.mean([f1 for (f1,_,_) in metrics[p_id]]) # avg of f1-scores
                                        if mean_score > max_score:
                                            max_score = mean_score
                                        elif mean_score < max_score - score_margin: # eliminate prematurely if the pipe performs really poorly
                                            rmvd_pipes[p_id] = True # eliminate this pipe prematurely (do not finish its cross-validation training)
                                            print('%i has been eliminated early: avg_score=%.2f and max_score=%.2f' % (p_id, mean_score, max_score))
                                
                        print('\n%i pipelines\' training have been stopped prematurely due to poor performances.' % len(rmvd_pipes)) # TODO tmp print
                        
                        print('\n3/3 - statistic tests') # TODO tmp print
                        # normalize runtime between 0 and 1 & compute the score of each pipeline on each cv-split
                        metrics = {p_id: val for p_id, val in metrics.items() if p_id not in rmvd_pipes}
                        max_runtime = max([i[2] for cv_scores in metrics.values() for i in cv_scores])
                        scores_ = {id: [ModelsTrainer._compute_selection_score(f1,r3,t/max_runtime, alpha, beta, gamma)
                                        for f1,r3,t in cv_scores] 
                                    for id, cv_scores in metrics.items()}
                        
                        # add the newly measured scores to the global scores list & remove the pipes that were stopped prematurely
                        for i in range(len(pipelines) - 1, -1, -1):
                            pipe = pipelines[i]
                            if pipe.id in scores_:
                                pipe.scores.extend(scores_[pipe.id])
                            else:
                                del pipelines[i]
                        
                        # statistical tests - remove pipes that are significantly worse than any other pipe
                        less_aggressive_pruning = (len(pipelines) <= get_max_nb_p_at_i(i)) if i > 0 else True
                        print('Using less aggressive pruning strategy: %s' % less_aggressive_pruning) # TODO tmp print
                        worse_pipes = self._apply_test(
                            pipelines, 
                            test_method,
                            K=pruning_factor,
                            p_value=p_value,
                            less_aggressive_pruning=less_aggressive_pruning
                        )

                        if len(pipelines) < 20: # TODO tmp print
                            print([(p.id, np.mean(p.scores)) for p in pipelines]) # TODO tmp print
                            print([
                                (test_method(a.scores, b.scores)[1], a.rm.id,b.rm.id)  # we keep the worse pipelines of the 2
                                    for a,b in itertools.combinations(pipelines, r=2) # for all pairs of pipes
                            ]) # TODO tmp print

                        # remove the worse pipes
                        pipelines = [p for p in pipelines if p not in worse_pipes]

                        print('\nThere remains %i pipelines. %i have been eliminated by t-test.' % (len(pipelines), len(worse_pipes)))

                        if early_break and len(pipelines) <= selection_len:
                            break

                # if we have more pipes remaining that what we wanted
                if len(pipelines) > selection_len: 
                    print('Too many pipelines remaining. Last attempt to eliminate "worse" candidates.')
                    pipelines = sorted(pipelines, key=lambda p: np.mean(p.scores), reverse=True)
                    for i in reversed(range(len(pipelines))): # rank pipes based on their avg scores
                        p_i = pipelines[i]

                        # pairwise ttest if p is worse than any other: prune p
                        for j, p_j in enumerate(pipelines):
                            if i is not j and test_method(p_i.scores, p_j.scores)[1] < p_value and np.mean(p_i.scores) < np.mean(p_j.scores):
                                print('%i was eliminated due to significantly worse performances than another candidate.' % p_i.id)
                                del pipelines[i]
                                break
                                
                        if len(pipelines) <= selection_len:
                            break

        except Exception as e:
            import logging
            logging.exception('Got exception while selecting pipelines.')
            if len(pipelines) >= init_nb_pipes:
                raise e
        return pipelines

    def train(self, models, train_on_all_data=False):
        """
        Trains and evaluates models given to this initialization of this trainer. Uses T-Daub for best-algorithms selection
        and cross-validation.

        Keyword arguments: 
        models -- list of RecommendationModel instances that should be trained and evaluated
        train_on_all_data -- True if the models should be trained on ALL data once the cross-val is done, False 
                             otherwise (default: False)

        Return:
        A TrainResults' instance containing the training results
        """
        self.models = models
        train_results = self._train(models, train_on_all_data=train_on_all_data)
        return train_results
        
    
    # private methods

    def _apply_test(self, pipelines, test_method, K=4, p_value=.01, less_aggressive_pruning=True):
        """
        Applies a significance test (paired t-test) to the pipelines to identify and return those that perform worse than others.

        Keyword arguments:
        pipelines -- list of ClfPipeline to apply t-test to.
        test_method -- Significance test. Takes as input two lists of equal length and returns a tuples which 2nd value is the p-value.
        K -- Pruning factor. Example: if K=4, the expected number of pipes to "survive" is 1/4 of the original set. (default: 4)
        p_value -- p_value used with the paired t-test to decide if a difference is significant or not (default: 0.01).
        less_aggressive_pruning -- uses a less aggressive pruning strategy if True, eliminates any pipe that is significantly worse than any other
                                   pipe otherwise. (default: True)

        Return:
        List of _TmpPipeline that are performing worse than others.
        """
        eliminated_pipes = []
        pipes_combinations = list(itertools.combinations(pipelines, r=2))
        worse_counters = {p: 0 for p in pipelines}
        
        T = len(pipelines) // K if less_aggressive_pruning else 0
        
        i = len(pipes_combinations)-1
        while i > 0:
            a,b = pipes_combinations[i]

            # if the paired t-test shows a statistical difference between the two
            if test_method(a.scores, b.scores)[1] < p_value:
                worse_pipe = a if np.mean(a.scores) < np.mean(b.scores) else b
                worse_counters[worse_pipe] += 1
                
                if worse_counters[worse_pipe] > T:
                    # eliminate the worst one
                    eliminated_pipes.append(worse_pipe)
                    # remove all pairs of pipes continaining the eliminated pipe to avoid unnecessary comparisons
                    for j in reversed(range(len(pipes_combinations))):
                        x,y = pipes_combinations[j]
                        if worse_pipe in (x,y):
                            del pipes_combinations[j]
                            i -= 1 if j < i else 0
            i -= 1
        return eliminated_pipes

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

                    #cm[0].savefig('Training/Results/Plots/%s: %i' % (model, split_id)) # TODO tmp print

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

    
    # static methods

    def _compute_selection_score(f1, r3, t, alpha, beta, gamma):
        """
        Method that computes the score based on which a model will be evaluated during the selection process. 

        Keyword arguments:
        f1 -- F1-score of a model measured on a test set
        r3 -- recall@3 of a model measured on a test set
        t -- model's training time
        alpha -- parameter value to define the impact of the F1-score in the score
        beta -- parameter value to define the impact of the Recall@3 in the score
        gamma -- parameter value to define the impact of the model's training time in the score

        Return:
        Returns the score of a model used during the selection process.
        """
        return ((alpha*f1) + (beta*r3) - (gamma*t) + gamma) / (alpha+beta+gamma)