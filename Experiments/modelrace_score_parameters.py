import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('../')

from recimpute import init_training_set, train, eval
from FeaturesExtraction.TSFreshFeaturesExtractor import TSFreshFeaturesExtractor
from FeaturesExtraction.TopologicalFeaturesExtractor import TopologicalFeaturesExtractor
from FeaturesExtraction.Catch22FeaturesExtractor import Catch22FeaturesExtractor
from FeaturesExtraction.KatsFeaturesExtractor import KatsFeaturesExtractor
from Labeling.ImputationTechniques.ImputeBenchLabeler import ImputeBenchLabeler
from Training.ClfPipeline import ClfPipeline
from Training.ModelsTrainer import ModelsTrainer
from Utils.Utils import Utils

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, Normalizer, QuantileTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import time
from tqdm import tqdm
import warnings

def custom_select(pipelines, training_set, MODELSTRAINER_CONF):

    print('#########  RecImpute - models\' selection  #########')

    _, all_pipelines_txt = ClfPipeline.generate(N=0)

    # most promising pipelines' selection
    trainer = ModelsTrainer(training_set)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        selected_pipes = trainer.select(
            pipelines, all_pipelines_txt, 
            S=MODELSTRAINER_CONF['S'], 
            selection_len=MODELSTRAINER_CONF['SELECTION_LEN'], 
            score_margin=MODELSTRAINER_CONF['SCORE_MARGIN'],
            n_splits=MODELSTRAINER_CONF['NB_CV_SPLITS'], 
            test_method=ttest_rel, 
            p_value=MODELSTRAINER_CONF['P_VALUE'],
            alpha=MODELSTRAINER_CONF['ALPHA'],
            beta=MODELSTRAINER_CONF['BETA'],
            gamma=MODELSTRAINER_CONF['GAMMA'],
            allow_early_eliminations=MODELSTRAINER_CONF['ALLOW_EARLY_ELIMINATIONS'],
            early_break=False,
        )
    return selected_pipes

def train_and_eval(pipelines, training_set, MODELSTRAINER_CONF):
    selected_pipes = custom_select(pipelines, training_set, MODELSTRAINER_CONF)
    tr, _, models = train([p.rm for p in selected_pipes], training_set, False)
    test_set = tr.load_set_from_archive('test')
    all_scores, _ = eval(models[1:], test_set, print_avg=False, print_details=False)
    return all_scores

def main():

    labeler = ImputeBenchLabeler.get_instance()
    labeler_properties = labeler.get_default_properties()
    true_labeler = true_labeler_properties = None
    features_extractors = [
        TSFreshFeaturesExtractor.get_instance(),
        TopologicalFeaturesExtractor.get_instance(),
        Catch22FeaturesExtractor.get_instance(),
        # KatsFeaturesExtractor.get_instance()
    ]
    
    training_set = init_training_set(labeler, labeler_properties, true_labeler, true_labeler_properties, features_extractors)

    pipelines = [
        ClfPipeline(0, make_pipeline(PCA(svd_solver='full'), BernoulliNB(alpha=1., fit_prior=True))),
        ClfPipeline(1, make_pipeline(KNeighborsClassifier(n_neighbors=5))),
        ClfPipeline(2, make_pipeline(Normalizer(norm='l1'), MaxAbsScaler(), PCA(svd_solver='full'), BernoulliNB(alpha=0.5))),
        ClfPipeline(3, make_pipeline(Normalizer(norm='l1'), MaxAbsScaler(), MLPClassifier(activation='logistic', alpha=1e-05, hidden_layer_sizes=250, learning_rate='adaptive', tol=0.01))),
        ClfPipeline(4, make_pipeline(CatBoostClassifier(verbose=False, loss_function='MultiClass'))),
        ClfPipeline(5, make_pipeline(Normalizer(), StandardScaler(), QuadraticDiscriminantAnalysis(tol=1e-2))),
        ClfPipeline(6, make_pipeline(PCA(svd_solver='auto'), RadiusNeighborsClassifier(radius=1.0, outlier_label='most_frequent'))),
        ClfPipeline(7, make_pipeline(Normalizer(), PCA(), CatBoostClassifier(verbose=False, loss_function='MultiClass'))),
        ClfPipeline(8, make_pipeline(Normalizer(), PCA(), GaussianNB(var_smoothing=1e-10))),
        ClfPipeline(9, make_pipeline(StandardScaler(), ExtraTreeClassifier(splitter='random', max_depth=5))),
        ClfPipeline(10, make_pipeline(Normalizer(), ExtraTreeClassifier(splitter='best', max_depth=15))),
        ClfPipeline(11, make_pipeline(LinearDiscriminantAnalysis(solver='svd', tol=1e-2))),
        ClfPipeline(12, make_pipeline(QuantileTransformer(), RadiusNeighborsClassifier(outlier_label='most_frequent'))),
        ClfPipeline(13, make_pipeline(QuantileTransformer(), CatBoostClassifier(verbose=False, learning_rate=0.01, iterations=30, loss_function='MultiClass'))),
        ClfPipeline(14, make_pipeline(QuantileTransformer(), PCA(svd_solver='full'), RandomForestClassifier(min_samples_leaf=5, n_estimators=50))),
        ClfPipeline(15, make_pipeline(PCA(svd_solver='arpack'), BernoulliNB(alpha=0.5))),
        ClfPipeline(16, make_pipeline(Normalizer(norm='l1'), PCA(svd_solver='arpack'), BernoulliNB(alpha=0.25, fit_prior=False))),
        ClfPipeline(17, make_pipeline(Normalizer(), QuadraticDiscriminantAnalysis(tol=0.01))),
        ClfPipeline(18, make_pipeline(PCA(), MLPClassifier(alpha=1e-05, hidden_layer_sizes=50, solver='sgd'))),
        ClfPipeline(19, make_pipeline(StandardScaler(), PCA(svd_solver='arpack'), QuadraticDiscriminantAnalysis())),
        ClfPipeline(20, make_pipeline(MaxAbsScaler(), PCA(svd_solver='full'), RadiusNeighborsClassifier(outlier_label='most_frequent', radius=50.0))),
        ClfPipeline(21, make_pipeline(Normalizer(), MLPClassifier(alpha=0.001, hidden_layer_sizes=50, learning_rate='invscaling', solver='sgd'))),
        ClfPipeline(22, make_pipeline(StandardScaler(), RadiusNeighborsClassifier(algorithm='ball_tree', outlier_label='most_frequent', radius=100.0))),
        ClfPipeline(23, make_pipeline(MaxAbsScaler(), PCA(svd_solver='randomized'), RadiusNeighborsClassifier(algorithm='brute', outlier_label='most_frequent', radius=100.0))	),
        ClfPipeline(24, make_pipeline(Normalizer(), StandardScaler(), PCA(svd_solver='arpack'), DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=10, splitter='random'))),
        ClfPipeline(25, make_pipeline(Normalizer(norm='l1'), MaxAbsScaler(), PCA(), RadiusNeighborsClassifier(algorithm='ball_tree', outlier_label='most_frequent'))),
        ClfPipeline(26, make_pipeline(Normalizer(norm='l1'), MaxAbsScaler(), PCA(svd_solver='full'), RadiusNeighborsClassifier(algorithm='brute', outlier_label='most_frequent'))),
        ClfPipeline(27, make_pipeline(Normalizer(norm='l1'), StandardScaler(), PCA(), DecisionTreeClassifier(max_depth=15, min_samples_leaf=2, min_samples_split=15))),
        ClfPipeline(28, make_pipeline(QuantileTransformer(), PCA(svd_solver='arpack'), MLPClassifier(activation='tanh', alpha=0.001, hidden_layer_sizes=100, learning_rate='adaptive', solver='sgd'))),
        ClfPipeline(29, make_pipeline(Normalizer(norm='l1'), QuantileTransformer(), PCA(svd_solver='arpack'), MLPClassifier(activation='tanh', alpha=1e-05, hidden_layer_sizes=50, solver='lbfgs', tol=0.001))),
        ClfPipeline(30, make_pipeline(MaxAbsScaler(), PCA(svd_solver='arpack'), LogisticRegression(C=10, multi_class='multinomial', tol=0.001))),
        ClfPipeline(31, make_pipeline(Normalizer(norm='l1'), PCA(svd_solver='arpack'), MLPClassifier(alpha=0.001, hidden_layer_sizes=250, learning_rate='adaptive'))),
        ClfPipeline(32, make_pipeline(Normalizer(norm='l1'), CatBoostClassifier(verbose=False, learning_rate=0.03, iterations=100, depth=10, loss_function='MultiClass'))),
        ClfPipeline(33, make_pipeline(Normalizer(norm='l1'), PCA(svd_solver='randomized'), LinearDiscriminantAnalysis(tol=0.01))),
        ClfPipeline(34, make_pipeline(Normalizer(norm='l1'), StandardScaler(), ExtraTreeClassifier(max_depth=15, min_samples_leaf=10, min_samples_split=10, splitter='best'))),
        ClfPipeline(34, make_pipeline(Normalizer(norm='l1'), StandardScaler(), ExtraTreeClassifier(max_depth=15, min_samples_leaf=10, min_samples_split=10, splitter='best'))),
    ]
    ClfPipeline.NEXT_PIPE_ID += len(pipelines)

    MODELSTRAINER_CONF = Utils.read_conf_file('modelstrainer')

    default_alpha, default_beta, default_gamma = .50, .50, .50
    alpha_values = [0., .25, .50, .75, 1.]
    beta_values = [0., .25, .50, .75, 1.]
    gamma_values = [0., .25, .50, .75, 1.]

    # vary alpha, then beta, and then gamma. measure accuracy metrics, and runtime.
    all_scores_per_param = {}
    for alpha in tqdm(alpha_values):
        MODELSTRAINER_CONF['ALPHA'] = alpha
        MODELSTRAINER_CONF['BETA'] = default_beta
        MODELSTRAINER_CONF['GAMMA'] = default_gamma
        start_time = time.time()
        all_scores_per_param['alpha=%.2f' % alpha] = train_and_eval(pipelines, training_set, MODELSTRAINER_CONF)
        all_scores_per_param['alpha=%.2f' % alpha]['Runtime'] = [time.time() - start_time]
    for beta in tqdm(beta_values):
        MODELSTRAINER_CONF['ALPHA'] = default_alpha
        MODELSTRAINER_CONF['BETA'] = beta
        MODELSTRAINER_CONF['GAMMA'] = default_gamma
        start_time = time.time()
        all_scores_per_param['beta=%.2f' % beta] = train_and_eval(pipelines, training_set, MODELSTRAINER_CONF)
        all_scores_per_param['beta=%.2f' % beta]['Runtime'] = [time.time() - start_time]
    for gamma in tqdm(gamma_values):
        MODELSTRAINER_CONF['ALPHA'] = default_alpha
        MODELSTRAINER_CONF['BETA'] = default_beta
        MODELSTRAINER_CONF['GAMMA'] = gamma
        start_time = time.time()
        all_scores_per_param['gamma=%.2f' % gamma] = train_and_eval(pipelines, training_set, MODELSTRAINER_CONF)
        all_scores_per_param['gamma=%.2f' % gamma]['Runtime'] = [time.time() - start_time]

    scores_df = pd.DataFrame(all_scores_per_param).applymap(lambda l: np.mean(l))
    print(scores_df.to_markdown())

if __name__ == '__main__':
    main()
