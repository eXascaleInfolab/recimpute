"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
AbstractLabeler.py
@author: @chacungu
"""

import abc

class AbstractLabeler(metaclass=abc.ABCMeta):
    """
    Abstract TODO
    """

    #CONF = Utils.read_conf_file('TODO')

    # create necessary directories if not there yet
    #if not os.path.exists(TODO_DIR):
    #    os.makedirs(TODO_DIR)


    """
    TODO 
    tsc_utils
        RAW_FEATURES_NAMES 
        RAW_FEATURES_THRESHOLD 
        LABELED_TIME_SERIES_APPENDIX 
        SYNTHETIC_LABELS_SUFIX 
        TRUE_LABELS_SUFIX 
        TRAINING_SETS_FOLDER 
        CLUSTERS_LABELS_FILENAME 
        _COMPLEX_FEATURES_TRAINING_SET_SUFFIX
        _RAW_FEATURES_TRAINING_SET_SUFFIX
        _TRUE_LABELS_TRAINING_SET_SUFFIX
        _SYNTHETIC_LABELS_TRAINING_SET_SUFFIX
        _MULTILABELS_TRAINING_SET_SUFFIX
        _REDUCED_LABELS_TRAINING_SET_SUFFIX
        _TRAINING_DATASETS_INFO
        ALGORITHMS_LABEL_KIVIAT 
        ALGORITHMS_LABEL 
        BENCHMARK_LABEL_STATUS_FILE 
        COMPLEX_FEATURES_CREATION_STATUS_FILE 

        map_kiviat_lbls_to_bchmrk
        map_labels_str_to_int
        map_labels_int_to_str
        get_labels_list_from_dataset
        get_training_dataset
        get_clusters_labels
        get_labels_filename
        get_dataset_labels
        _get_training_set_id
        get_training_set_info
        get_training_set_filename
        get_validation_set_filename
        get_set_info_filename
        _get_reduced_labels_perc_from_filename
        get_labels_list

    benchmark
        BENCHMARK_PATH

        run                             x   _run_benchmark
        get_highest_benchmark_score
        get_result_plots

    training_sets_creation
        recov_algos_weights 
        weights 

        get_irregularity_score
        compute_raw_features
        extract_features_for_kiviat
        apply_kiviat_rules
        propagate_cluster_labels
        label_clusters_with_kiviat
        _get_best_algo                      x
        _create_reduced_multilabels_set     x
        _create_multilabels_vector          x
        _label_clusters_with_benchmark      x   _label_cluster
        _label_from_benchmark_results       x
        label_clusters_with_benchmark       x   label
        create_algos_score_matrix           x
        create_algos_ranking_matrix         x
        _splitter_func
        get_dataset_features
        _create_complex_training_subset
        create_complex_training_set
        create_raw_training_set
    """


    # public methods

    @abc.abstractmethod
    def label(self, datasets):
        pass


    # private methods
    
    # self


    # static methods

    @abc.abstractmethod
    def load_labels(dataset_name, properties):
        pass

    @abc.abstractmethod
    def save_labels(dataset_name, labels):
        pass