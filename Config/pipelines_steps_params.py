"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
pipelines_steps_params.py
@author: @chacungu
"""

from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler, Normalizer, QuantileTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# This files lists the steps that a Training.ClfPipeline can be made of and their range of possible parameters.

ALL_STEPS = {
    'normalizers': {
        None: {},
        Normalizer: {
            'norm': ['l1', 'l2'],
        },
    },
    'scalers': {
        None: {},
        MaxAbsScaler: {}, 
        QuantileTransformer: {}, 
        StandardScaler: {}, 
    },
    'dim_reduction': {
        None: {},
        PCA: {
            'svd_solver': ['auto', 'full', 'arpack', 'randomized'],
        },
    },
    'classifiers': {
        KNeighborsClassifier: {
            'n_neighbors': [1, 3, 5, 10],
            'weights': ['uniform', 'distance'],
        },
        CatBoostClassifier: {
            'depth': [4,5,6,7,8,9,10],
            'learning_rate': [0.01,0.02,0.03,0.04],
            'iterations': [10, 20,30,40,50,60,70,80,90, 100],
            'loss_function': ['MultiClass'],
            'verbose': [False],
        },
        RandomForestClassifier: {
            'n_estimators': [10, 50, 200, 500, 1000],
            'max_depth': [5, 8, 15, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10],
        },
        SVC: {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': [1e-4, 1e-3, 'auto'],
            'C': [1, 10, 100, 1000],
            'tol': [1e-4, 1e-3],
        },
    },
}