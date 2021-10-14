"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
maxabsscaler_catboostclassifier.py
@author: @chacungu
"""

# Please see the ./_template.py file if you wish to add your own model description.

# Description of a model made of a Scikit-Learn' MaxAbstScaler and a CatBoost' classifier.

from catboost import CatBoostClassifier
from scipy.stats import uniform, randint
from sklearn.preprocessing import MaxAbsScaler

model_info = {
    
    # ---------------------------------------------------------------
    # MANDATORY FIELDS:

    'name': 'maxabsscaler_catboostclassifier',

    ## List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, 
    ## with the last object an estimator. Transform should be a class (not an instance!). 
    'steps': [
        ('MaxAbsScaler', MaxAbsScaler),
        ('CatBoostClassifier', CatBoostClassifier),
    ],

    ## Dictionary defining the ranges of possible values of each step's parameters.
    'params_ranges': {
        'CatBoostClassifier__depth': randint(4, 10),
        'CatBoostClassifier__learning_rate' : uniform(),
        'CatBoostClassifier__iterations': randint(10, 100),
        'CatBoostClassifier__loss_function': ['MultiClass'],
        'CatBoostClassifier__verbose': [False],
    },

    # Integer between 1 (fast) and 3 (slow) used to indicate the expected training speed of this model.
    'training_speed_factor': 2,

    # ---------------------------------------------------------------
    # OPTIONAL FIELDS (if set to None, these fields will not be used)

    'multiclass_strategy': None,

    'bagging_strategy': None,
    
}