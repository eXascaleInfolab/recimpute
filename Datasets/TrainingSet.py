"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
TrainingSet.py
@author: @chacungu
"""

import warnings

from Datasets.Dataset import Dataset
from Utils.Utils import Utils

class TrainingSet:
    """
    Class which handles a training set to be used in classification / regression tasks.
    """

    CONF = Utils.read_conf_file('trainingset')


    # constructor

    def __init__(self, datasets, clusterer, features_extracters, labeler, true_labeler=None, force_generation=False):
        """
        Initializes a TrainingSet object.

        Keyword arguments:
        datasets -- list of Dataset objects
        clusterer -- instance of a clusterer
        features_extracters -- list of instances of features extracters
        labeler -- instance of a labeler used to label training set
        true_labeler -- instance of a "true" labeler used to label only the test set (default None: use the labeler)
        force_generation -- True if the clusters, labels, and features must be created even if they already exist, False otherwise
        """
        self.clusterer = clusterer
        self.labeler = labeler
        self.true_labeler = true_labeler
        self.features_extracters = features_extracters # list
        
        # make sure clustering has been done (and do it otherwise)
        updated_datasets, clusters_created = self.__init_clustering(datasets, force_generation)

        # reserve some data for testing
        self.test_set_level, self.test_set_ids = self.__init_test_set(datasets, 
                                                                      TrainingSet.CONF['TEST_SET_RESERVATION_STRAT'])

        # make sure labeling & features extraction has been done (and do it otherwise)
        updated_datasets = self.__init_labeling_and_fe(updated_datasets, force_generation | clusters_created)
        self.datasets = updated_datasets

    def __init_test_set(self, datasets, strategy):
        """
        Reserves some data for testing according to the specified strategy.

        Keyword arguments:
        datasets -- list of Dataset objects
        strategy -- string specifying the test set reservation strategy
        
        Return:
        1. string specifying the level at which data is reserved (clusters or datasets)
        2. list of ids to identify the data reserved for testing
        """
        if strategy == 'one_cluster_every_two_dataset':
            # reserve one cluster every two data sets for the test set
            test_set = []
            for dataset in datasets[::2]:
                for cid in dataset.cids:
                    test_set.append(cid)
                    break
            return 'clusters', test_set
        else:
            raise Exception('Test set reservation strategy not implemented: ', strategy)

    def __init_clustering(self, datasets, force_generation):
        """
        Makes sure each data set's clustering has already been done. If not, finds the missing clusters.

        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        force_generation -- boolean to indicate if the clusters must be generated even if they already exist
        
        Return:
        1. List of Dataset objects
        2. True if some data set's were missing clusters and must have been clustered, False otherwise
        """
        clusters_generated = False
        updated_datasets = []
        for dataset in datasets:
            if not dataset.are_clusters_created():
                # create clusters if they have not been yet created for this data set
                if not clusters_generated:
                    warnings.warn('Some data set\'s time series have not been clustered yet. \
                                    They will be clustered now. Clustering data sets one-by-one is \
                                    less-efficient than using the clusterer\'s method "cluster_all_datasets" \
                                    before the instantiation of a TrainingSet object.')
                dataset = self.clusterer.cluster(dataset)
                clusters_generated = True
            updated_datasets.append(dataset)
        if clusters_generated:
            # merge clusters with <5 time series to the most similar cluster from the same data set
            updated_datasets = self.clusterer.merge_small_clusters(updated_datasets, min_nb_ts=self.clusterer.CONF['MIN_NB_TS_PER_CLUSTER'])
        if clusters_generated or not self.clusterer.are_cids_unique(updated_datasets):
            # change all clusters' ID (for all datasets) such that there are no duplicates
            updated_datasets = self.clusterer.make_cids_unique(updated_datasets)

        return updated_datasets, clusters_generated

    def __init_labeling_and_fe(self, datasets, force_generation):
        """
        Makes sure each data set's labels and features have already been created. If not, creates the missing ones.

        Keyword arguments:
        datasets -- list of Dataset objects
        force_generation -- boolean to indicate if the labels & features must be generated even if they already exist
        
        Return:
        List of Dataset objects
        """
        updated_datasets = []
        for dataset in datasets:
            # labeling
            if force_generation or not self.labeler.are_labels_created(dataset.name):
                # create labels if they have not been yet created for this data set
                dataset = self.labeler.label(dataset)
            if self.true_labeler is not None and self.is_in_test_set(dataset): 
                if force_generation or not self.true_labeler.are_labels_created(dataset.name):
                    # create true labels if they have not been yet created for this data set
                    dataset = self.true_labeler.label(dataset)

            # features extraction
            for features_extracter in self.features_extracters:
                if force_generation or not features_extracter.are_features_created(dataset.name):
                    # create features if they have not been yet created for this data set
                    dataset = features_extracter.extract(dataset)
            updated_datasets.append(dataset)

        return updated_datasets


    # public methods

    def get_reduced_training_set(self, train, reduction_perc):
        # TODO
        # sklearn_train_test_split
        # returns train_reduced
        pass

    def get_balanced_training_set(self, train, according_to):
        # TODO
        # train is df with extra info cols (such as Cluster ID and Time Series ID) before being splitted into X_train and y_train
        # according_to can be one of: 'clusters', or 'labels'
        # returns train_balanced
        pass

    def yield_splitted_train_and_val(properties):
        # TODO
        # call inside cv loop
        # properties: e.g. augment=True, balance='clusters'
        # split_train_test_sets
        # yields X_train, y_train, X_val, y_val
        pass

    def augment_train(X_train, y_train):
        # TODO
        # returns X_train_augmented, y_train_augmented
        pass

    def is_in_test_set(self, dataset):
        """
        Checks whether any of the data set's cluster or the whole data set are in the test set.

        Keyword arguments:
        dataset -- Dataset object
        
        Return:
        True if any of the data set's cluster or the whole data set are in the test set, False otherwise
        """
        if self.test_set_level == 'clusters':
            return any(cid in self.test_set_ids for cid in dataset.cids)
        elif self.test_set_level == 'datasets':
            return dataset.name in self.test_set_ids
        else:
            raise Exception('Test set reservation strategy not implemented: ', TrainingSet.CONF['TEST_SET_RESERVATION_STRAT'])