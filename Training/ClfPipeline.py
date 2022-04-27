"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
ClfPipeline.py
@author: @chacungu
"""

import itertools
import numpy as np
import random as rdm
from sklearn.pipeline import make_pipeline

from Training.RecommendationModel import RecommendationModel
from Config.pipelines_steps_params import ALL_STEPS

rdm.seed(12345)

class ClfPipeline:
    """
    Class which handles a classification pipeline using Scikit-Learn's pipelines. This class is used only during the selection process.
    """

    CLF_IDS_DICT = {clf: i for i,clf in enumerate(list(ALL_STEPS['classifiers'].keys()))}
    NEXT_PIPE_ID = 0


    # constructor
    def __init__(self, id, pipe):
        self.id = id
        self.rm = RecommendationModel(id, 'classifier', pipe)
        self.scores = []

    def __repr__(self):
        return '%s: %.5f' % (self.rm, np.mean(self.scores))

    def has_same_steps(self, pipe2_text):
        """
        Compares this object's pipeline against the pipeline's textual description given in parameter.
        
        Keyword arguments: 
        pipe2_text -- pipeline's textual description to compare to

        Return
        True if the two pipelines have the same steps and parameters' values, False otherwise.
        """
        pipe2_steps = ClfPipeline.make_steps(pipe2_text)
        for i, (step_name, step) in enumerate(self.rm.pipe.steps[:-1]):
            if not isinstance(pipe2_steps[i], step.__class__):
                return False
        return True


    # static methods
    
    def generate(N):
        """
        Generates N new random pipelines.

        Keyword arguments:
        N -- numbers of pipelines to generate

        Return:
        1. list of ClfPipeline objects
        2. list of sets containing all possible combinations of pipelines (search space)
        """
        def _get_all(all_x):
            # creates all combinations of steps and params
            res = []
            for x, params in all_x.items():
                if x is not None:
                    values = [
                        list(itertools.product([key], values)) 
                        for key, values in params.items()
                    ]
                else:
                    values = [[None]]
                res.extend(itertools.product([x], *values))
            return res

        all_pre_pipelines = list(itertools.product(
            *[_get_all(ALL_STEPS[step_name]) for step_name in ALL_STEPS.keys() if step_name != 'classifiers']
        ))
        all_pipelines = []
        for clf in ALL_STEPS['classifiers'].keys():
            all_pipelines.append(set(
                itertools.product(
                    all_pre_pipelines, 
                    itertools.product([clf], *[list(itertools.product([key], values)) for key, values in ALL_STEPS['classifiers'][clf].items()]))
            ))

        pipelines_text, all_pipelines = ClfPipeline._cstm_sample(all_pipelines, N)
                        
        pipelines = [
            ClfPipeline(id= i, pipe= make_pipeline(*ClfPipeline.make_steps(pipe_t)))
            for i, pipe_t in enumerate(pipelines_text)
        ]

        ClfPipeline.NEXT_PIPE_ID += len(pipelines)
        return pipelines, all_pipelines

    def generate_from_set(pipelines, all_pipelines_txt, nb_new_pipes_total):
        """
        Generates new pipelines from the existing set.

        Keyword arguments:
        pipelines -- list of ClfPipeline to use as "parents" for the new pipelines to sample
        all_pipelines_txt -- list of sets containing all possible combinations of pipelines (remaining search space)
        nb_new_pipes_total -- number of new pipelines to generate

        Return:
        1. list of ClfPipeline objects
        2. list of sets containing all possible combinations of pipelines (search space)
        """
        def _add_new_pipes(new_pipes_text, new_pipes):
            new_pipes.extend(
                map(
                    lambda x: ClfPipeline(
                        ClfPipeline.NEXT_PIPE_ID + x[0], 
                        make_pipeline(*ClfPipeline.make_steps(x[1]))), 
                    enumerate(new_pipes_text) 
                )
            )
            ClfPipeline.NEXT_PIPE_ID += len(new_pipes_text)
            return new_pipes
        nb_new_pipes_per_type = (2*nb_new_pipes_total//3) // len(pipelines)
        nb_new_rdm_pipes = nb_new_pipes_total - nb_new_pipes_per_type
        new_pipes = []
        for pipe in pipelines:
            clf_id = ClfPipeline.CLF_IDS_DICT[pipe.rm.pipe[-1].__class__]
            same_steps_pipes = {
                p_text for p_text in all_pipelines_txt[clf_id] 
                if pipe.has_same_steps(p_text)
            }
            new_pipes_text = rdm.sample(same_steps_pipes, min(nb_new_pipes_per_type, len(same_steps_pipes)))
            all_pipelines_txt[clf_id].difference_update(new_pipes_text)
            new_pipes = _add_new_pipes(new_pipes_text, new_pipes)

        new_pipes_text, all_pipelines_txt = ClfPipeline._cstm_sample(all_pipelines_txt, nb_new_rdm_pipes)
        new_pipes = _add_new_pipes(new_pipes_text, new_pipes)
        
        return new_pipes

    def make_steps(pipe_text):
        """
        Creates pipelines steps from its textual description.

        Keyword arguments:
        pipe_text -- textual description of the pipeline to create.

        Return:
        list of actual pipeline steps.
        """
        def _add_step(elem, steps):
            if elem[0] is not None:
                x = elem[0]
                params = {t[0]: t[1] for t in elem[1:]}
                steps.append(x(**params))
            return steps

        steps = []
        for elem in pipe_text[0]:
            steps = _add_step(elem, steps)

        steps = _add_step(pipe_text[1], steps)
        return steps
        
    
    # private methods

    def _cstm_sample(all_pipes, n):
        """
        TODO
        """
        selection = []
        for _ in range(n):        
            clf_id = rdm.randint(0, len(all_pipes)-1) # randomly select a type of classifier
            pipe = rdm.sample(all_pipes[clf_id], 1)[0] # randomly select a pipe which head is of the selected clf type
            all_pipes[clf_id].remove(pipe) # remove the pipe from the set of still available pipelines
            selection.append(pipe)
            
            if len(all_pipes[clf_id]) <= 0:
                del all_pipes[clf_id]
        return selection, all_pipes