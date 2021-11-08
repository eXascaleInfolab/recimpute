"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
kneighbors.py
@author: @chacungu
"""

# Please see the ./_template.py file if you wish to add your own model description.

# Description of a model made of a Scikit-Learn' k-Neighbors classifier.

from sklearn.neighbors import KNeighborsClassifier

model_info = {
    
    # ---------------------------------------------------------------
    # MANDATORY FIELDS:

    'name': 'kneighbors',
    'type': 'classifier',

    ## List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, 
    ## with the last object an estimator. Transform should be a class (not an instance!). 
    'steps': [
        ('KNeighborsClassifier', KNeighborsClassifier),
    ],

    ## Dictionary defining the ranges of possible values of each step's parameters.
    'params_ranges': {
        'KNeighborsClassifier__n_neighbors': [1, 3, 5, 10],
        'KNeighborsClassifier__weights': ['uniform', 'distance'],
        'KNeighborsClassifier__n_jobs': [-1],
    },

    # Integer between 1 (fast) and 3 (slow) used to indicate the expected training speed of this model.
    'training_speed_factor': 2,

    # ---------------------------------------------------------------
    # OPTIONAL FIELDS (if set to None, these fields will not be used)

    'multiclass_strategy': None,

    'bagging_strategy': None,
    
}