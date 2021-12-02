"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
standardscaler_svc.py
@author: @chacungu
"""

# Please see the ./_template.py file if you wish to add your own model description.

# Description of a model made of a Scikit-Learn' Standard Scaler and a Scikit-Learn' SVC

from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model_info = {
    
    # ---------------------------------------------------------------
    # MANDATORY FIELDS:

    'name': 'standardscaler_svc',
    'type': 'classifier',

    ## List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, 
    ## with the last object an estimator. Transform should be a class (not an instance!). 
    'steps': [
        ('StandardScaler', StandardScaler),
        ('SVC', SVC),
    ],

    ## Dictionary defining the ranges of possible values of each step's parameters.
    'params_ranges': {
        'SVC__kernel': ['rbf', 'poly', 'sigmoid'],
        'SVC__gamma': [1e-4, 1e-3, 'auto'],
        'SVC__C': [1, 10, 100, 1000],
        'SVC__tol': [1e-4, 1e-3],
    },

    ## Dictionary of the default parameter names mapped to their values. Can be set to None if the default parameters should be
    ## the model's default parameters.
    'default_params': None,

    # Integer between 1 (fast) and 3 (slow) used to indicate the expected training speed of this model.
    'training_speed_factor': 3,

    # ---------------------------------------------------------------
    # OPTIONAL FIELDS (if set to None, these fields will not be used)

    'multiclass_strategy': OneVsRestClassifier, 

    'bagging_strategy': BaggingClassifier,
    
}