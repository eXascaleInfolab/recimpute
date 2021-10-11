"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
normalizer_randomforest.py
@author: @chacungu
"""

# Please see the ./_template.py file if you wish to add your own model description.

# Description of a model made of a Scikit-Learn' Normalizer and a Scikit-Learn' Random Forest.

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

model_info = {
    
    # ---------------------------------------------------------------
    # MANDATORY FIELDS:

    'name': 'normalizer_randomforest',

    ## Ordered list of transformations and operations applied to the data. Should end with a classification or regression model.
    ## Should be a list of classes (not instances!).
    'steps': [
        Normalizer,
        RandomForestClassifier,
    ],

    ## Dictionary defining the ranges of possible values of each step's parameters.
    'params_range': {
        'Normalizer__norm': ['l1', 'l2'],

        'RandomForestClassifier__n_estimators': [10, 50, 200, 500, 1000],
        'RandomForestClassifier__max_depth': [5, 8, 15, None],
        'RandomForestClassifier__min_samples_split': [2, 5, 10, 15],
        'RandomForestClassifier__min_samples_leaf': [1, 2, 5, 10],
        'RandomForestClassifier__n_jobs': [-1],
    },

    # ---------------------------------------------------------------
    # OPTIONAL FIELDS (if set to None, these fields will not be used)

    'multiclass_strategy': None, # CHANGE THIS

    'bagging_strategy': None, # CHANGE THIS
    
}