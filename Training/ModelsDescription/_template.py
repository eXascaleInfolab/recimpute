"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
_template.py
@author: @chacungu
"""

# THIS FILE IS A TEMPLATE TO SIMPLIFY THE ADDITION OF NEW MODELS TO THE SYSTEM
# DO NOT USE THIS FILE AS A MODEL SINCE THIS IS ONLY A TEMPLATE

model_info = {
    
    # ---------------------------------------------------------------
    # MANDATORY FIELDS:

    # name of this model
    'name': 'normalizer_randomforest',

    ## List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, 
    ## with the last object an estimator. Name should be the transform class' name. Transform should be a class (not an instance!). 
    ## Example: 'RandomForestClassifier': sklearn.ensemble.RandomForestClassifier.
    'steps': [
        ('None', None), # CHANGE THIS
    ],

    ## Dictionary defining the ranges of possible values of each step's parameters. Each parameter's key is the concatenation of
    ## the step's name (class name), two underscores, and the parameter's name.
    ## Example: our last step is a RandomForestClassifier and we want to set the range for its "max_depth" parameter. The entry
    ## in the dict below could be something like: "RandomForestClassifier__max_depth": [5, 8, 15],
    'params_ranges': {
        'None': None, # CHANGE THIS
    },

    # Integer between 1 (fast) and 3 (slow) used to indicate the expected training speed of this model. The slowest, the less
    # gridsearch iterations will be done.
    'training_speed_factor': None, # CHANGE THIS

    # ---------------------------------------------------------------
    # OPTIONAL FIELDS (if set to None, these fields will not be used)

    ## multiclass strategy
    'multiclass_strategy': None, # CHANGE THIS

    ## bagging strategy
    'bagging_strategy': None, # CHANGE THIS
    
}