import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('../')

from recimpute import init_training_set, select, train
from FeaturesExtraction.TSFreshFeaturesExtractor import TSFreshFeaturesExtractor
from FeaturesExtraction.TopologicalFeaturesExtractor import TopologicalFeaturesExtractor
from FeaturesExtraction.Catch22FeaturesExtractor import Catch22FeaturesExtractor
from FeaturesExtraction.KatsFeaturesExtractor import KatsFeaturesExtractor
from Labeling.ImputationTechniques.ImputeBenchLabeler import ImputeBenchLabeler
from Utils.Utils import Utils

from tqdm import tqdm

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

    MODELSTRAINER_CONF = Utils.read_conf_file('modelstrainer')

    all_config_values = [50, 100, 300, 500, 700, 900, 1000]

    # vary nb sampled pipelines
    for nb_sampled_pipelines in tqdm(all_config_values):
        MODELSTRAINER_CONF['NB_PIPELINES'] = nb_sampled_pipelines
        print('~~ Starting selection with #sampled pipelines = %i ~~' % nb_sampled_pipelines)
        select(training_set, MODELSTRAINER_CONF)
    print('Done')

if __name__ == '__main__':
    main()