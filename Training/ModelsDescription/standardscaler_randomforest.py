"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
standardscaler_randomforest.py
@author: @chacungu
"""

# Please see the ./_template.py file if you wish to add your own model description.

# Description of a model made of a Scikit-Learn' StandardScaler and a Scikit-Learn' Random Forest.

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

model_info = {
    
    # ---------------------------------------------------------------
    # MANDATORY FIELDS:

    'name': 'standardscaler_randomforest',
    'type': 'classifier',

    ## List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, 
    ## with the last object an estimator. Transform should be a class (not an instance!). 
    'steps': [
        ('StandardScaler', StandardScaler),
        ('RandomForestClassifier', RandomForestClassifier),
    ],

    ## Dictionary defining the ranges of possible values of each step's parameters.
    'params_ranges': {
        'RandomForestClassifier__n_estimators': [10, 50, 200, 500, 1000],
        'RandomForestClassifier__max_depth': [5, 8, 15, None],
        'RandomForestClassifier__min_samples_split': [2, 5, 10, 15],
        'RandomForestClassifier__min_samples_leaf': [1, 2, 5, 10],
        'RandomForestClassifier__n_jobs': [-1],
    },

    # Integer between 1 (fast) and 3 (slow) used to indicate the expected training speed of this model.
    'training_speed_factor': 2,

    # ---------------------------------------------------------------
    # OPTIONAL FIELDS (if set to None, these fields will not be used)

    'multiclass_strategy': None,

    'bagging_strategy': None,
    
}