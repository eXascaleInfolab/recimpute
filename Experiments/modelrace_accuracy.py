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
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler, Normalizer, QuantileTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.pipeline import make_pipeline
import warnings

def print_scores(models, tr):
    test_set = tr.load_set_from_archive('test')
    all_scores, _ = eval(models[1:], test_set, print_avg=False, print_details=False)
    all_scores_df = pd.DataFrame(all_scores)
    print('Minimum scores')
    print(all_scores_df.min())
    print('Maximum scores')
    print(all_scores_df.max())
    print('Average scores')
    print(all_scores_df.mean())

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

def main():

    labeler = ImputeBenchLabeler.get_instance()
    labeler_properties = labeler.get_default_properties()
    true_labeler = true_labeler_properties = None
    features_extractors = [
        TSFreshFeaturesExtractor.get_instance(),
        TopologicalFeaturesExtractor.get_instance(),
        Catch22FeaturesExtractor.get_instance(),
        KatsFeaturesExtractor.get_instance()
    ]
    
    training_set = init_training_set(labeler, labeler_properties, true_labeler, true_labeler_properties, features_extractors)

    pipelines = [
        ClfPipeline(0, make_pipeline(PCA(svd_solver='full'), BernoulliNB(alpha=1., fit_prior=True))),
        ClfPipeline(1, make_pipeline(KNeighborsClassifier(n_neighbors=5))),
        ClfPipeline(2, make_pipeline(Normalizer(norm='l1'), MaxAbsScaler(), PCA(svd_solver='full'), BernoulliNB(alpha=0.5))),
        ClfPipeline(3, make_pipeline(Normalizer(norm='l1'), MaxAbsScaler(), MLPClassifier(activation='logistic', alpha=1e-05, hidden_layer_sizes=250, learning_rate='adaptive', tol=0.01))),
        ClfPipeline(4, make_pipeline(CatBoostClassifier(verbose=False))),
        ClfPipeline(5, make_pipeline(Normalizer(), StandardScaler(), QuadraticDiscriminantAnalysis(tol=1e-2))),
        ClfPipeline(6, make_pipeline(PCA(svd_solver='auto'), RadiusNeighborsClassifier(radius=1.0, outlier_label='most_frequent'))),
        ClfPipeline(7, make_pipeline(Normalizer(), PCA(), CatBoostClassifier(verbose=False))),
        ClfPipeline(8, make_pipeline(Normalizer(), PCA(), GaussianNB(var_smoothing=1e-10))),
        ClfPipeline(9, make_pipeline(StandardScaler(), ExtraTreeClassifier(splitter='random', max_depth=5))),
        ClfPipeline(10, make_pipeline(Normalizer(), ExtraTreeClassifier(splitter='best', max_depth=15))),
        ClfPipeline(11, make_pipeline(LinearDiscriminantAnalysis(solver='svd', tol=1e-2))),
        ClfPipeline(12, make_pipeline(QuantileTransformer(), RadiusNeighborsClassifier(outlier_label='most_frequent'))),
        ClfPipeline(13, make_pipeline(QuantileTransformer(), CatBoostClassifier(verbose=False, learning_rate=0.1, iterations=30))),
        ClfPipeline(14, make_pipeline(QuantileTransformer(), PCA(svd_solver='full'), RandomForestClassifier(min_samples_leaf=5, n_estimators=50))),
    ]
    ClfPipeline.NEXT_PIPE_ID += len(pipelines)

    # first: evaluate the set of pipelines without applying ModelRace
    tr, _, models = train([p.rm for p in pipelines], training_set, False)
    

    # then apply ModelRace to only keep the most-promising pipelines and see if it improves the min, max, and average metrics values
    MODELSTRAINER_CONF = Utils.read_conf_file('modelstrainer')
    MODELSTRAINER_CONF['SELECTION_LEN'] = 5
    selected_pipes = custom_select(pipelines, training_set, MODELSTRAINER_CONF)
    tr2, _, models2 = train([p.rm for p in selected_pipes], training_set, False)

    print('\n\n Scores comparison:')
    print('~~ Without any selection: ~~')
    print_scores(models, tr)
    print('~~ After using ModelRace: ~~')
    print_scores(models2, tr2)


if __name__ == '__main__':
    main()