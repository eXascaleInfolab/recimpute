"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
pipelines_steps_params.py
@author: @chacungu
"""

from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler, Normalizer, QuantileTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

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
            'n_neighbors': [1, 3, 5, 10, 15, 25, 50, 100, 200],
            'weights': ['uniform', 'distance'],
        },
        CatBoostClassifier: {
            'depth': [4,5,6,7,8,9,10],
            'learning_rate': [0.01,0.02,0.03,0.04],
            'iterations': [10,20,30,40,50,60,70,80,90,100],
            'loss_function': ['MultiClass'],
            'verbose': [False],
        },
        RandomForestClassifier: {
            'n_estimators': [10, 50, 200],
            'max_depth': [5, 8, 15, 20],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10],
        },

        BernoulliNB: {
            'alpha': [0., .25, .50, .75, 1.],
            'fit_prior': [True, False],            
        },
        DecisionTreeClassifier: {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [5, 8, 15, 20],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10],
        },
        ExtraTreeClassifier: {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [5, 8, 15, 20],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 5, 10],
        },
        GaussianNB: {
            'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
        },
        LinearDiscriminantAnalysis: {
            'solver': ['svd', 'lsqr'],
            'tol': [1e-2,1e-3,1e-4,1e-5],
        },
        LogisticRegression: {
            'multi_class': ['multinomial'],
            'solver': ['newton-cg', 'sag', 'saga', 'lbfgs'],
            'tol': [1e-2,1e-3,1e-4,1e-5],
            'C': [.1, 1, 10, 100],
        },
        MLPClassifier: {
            'hidden_layer_sizes': [50, 100, 250],
            'activation': ['logistic', 'relu', 'tanh'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [1e-3, 1e-4, 1e-5],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'tol': [1e-2,1e-3,1e-4,1e-5],
        },
        QuadraticDiscriminantAnalysis: {
            'tol': [1e-2,1e-3,1e-4,1e-5],
        },
        RadiusNeighborsClassifier: {
            'radius': [1.0, 10., 50., 100.],
            'outlier_label': ['most_frequent'],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        },
        
        # RidgeClassifier: {
        #     'normalize': [True, False],
        #     'tol': [1e-2,1e-3,1e-4],
        #     'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        # },
        # LinearSVC: {
        #     'multi_class': ['crammer_singer'],
        #     'tol': [1e-2,1e-3,1e-4,1e-5],
        #     'C': [.1, 1, 10, 100],
        # },
        # LabelPropagation: {
        #     'kernel': ['knn', 'rbf'],
        #     'gamma': [10,15,20,25],
        #     'n_neighbors': [1, 3, 5, 10, 15, 25, 50, 100, 200],
        #     'tol': [1e-2,1e-3,1e-4],
        # },
        # LabelSpreading: {
        #     'kernel': ['knn', 'rbf'],
        #     'gamma': [10,15,20,25],
        #     'n_neighbors': [1, 3, 5, 10, 15, 25, 50, 100, 200],
        #     'alpha': [.1,.2,.4,.6,.8],
        #     'tol': [1e-2,1e-3,1e-4],
        # },
        #SVC: {
        #    'kernel': ['rbf', 'poly', 'sigmoid'],
        #    'gamma': [1e-4, 1e-3, 'auto'],
        #    'C': [1, 10, 100, 1000],
        #    'tol': [1e-4, 1e-3],
        #},
    },
}