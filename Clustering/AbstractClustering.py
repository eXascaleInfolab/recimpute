"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
AbstractClustering.py
@author: @chacungu
"""

import abc
from kshape.core import _ncc_c
import numpy as np
import numpy.core.numeric as _nx
import operator
from os.path import isfile, normpath as normp
import pandas as pd

from Datasets.Dataset import Dataset
from Utils.Utils import Utils
from Utils.SingletonClass import SingletonClass

class AbstractClustering(SingletonClass, metaclass=abc.ABCMeta):
    """
    Abstract Clustering class used to cluster time series and handle those clusters.
    """
    
    CLUSTERS_DIR = normp('./Clustering/cassignments/')
    CLUSTERS_APPENDIX = '_cassignments.csv'

    # create necessary directories if not there yet
    Utils.create_dirs_if_not_exist([CLUSTERS_DIR])
    

    @abc.abstractmethod
    def save_clusters(self, dataset, cassignment):
        pass

    @abc.abstractmethod
    def load_clusters(self, dataset_name):
        pass
    
    @abc.abstractmethod
    def cluster_all_datasets(self, datasets):
        pass

    @abc.abstractmethod
    def cluster(self, dataset):
        pass

    @abc.abstractmethod
    def _get_cassignment_filename(self, dataset_name):
        pass

    def are_clusters_created(self, dataset_name):
        """
        Checks whether the clusters exist or not.
        
        Keyword arguments: -
        
        Return: 
        True if the clusters have already been created and saved as CSV, false otherwise.
        """
        cassignment_filename = self._get_cassignment_filename(dataset_name)
        return isfile(cassignment_filename)

    def make_cids_unique(self, datasets):
        """
        Iterates over all Datasets and updates their clusters' ID such that they are unique.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        
        Return:
        List of Dataset objects containing the time series.
        """
        next_global_cid = 0
        updated_datasets = []
        for dataset in datasets:
            # load clusters assignment of each dataset
            clusters_assignment = self.load_clusters(dataset.name)
            # update cluster ids such that there are no duplicates over all datasets
            for i, cluster_id in enumerate(clusters_assignment['Cluster ID'].unique()):
                clusters_assignment['Cluster ID'].replace(cluster_id, '#' + str(i + next_global_cid), inplace=True)
            clusters_assignment['Cluster ID'] = clusters_assignment['Cluster ID'].map(lambda v: int(v.replace('#', '')))
            next_global_cid += len(clusters_assignment['Cluster ID'].unique())
            # save modified assignments to csv
            self.save_clusters(dataset, clusters_assignment)
            updated_datasets.append(dataset)
        return updated_datasets

    def apply_constraints(self, datasets, min_nb_ts, max_nb_ts):
        """
        Applies clusters' constraints. Each should have between 5 and 15 (variables defined in config file) time series.
        Updates the clusters assignment csv files.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        min_nb_ts -- minimum number of time series a cluster can have without requiring a merge.
        max_nb_ts -- maximum number of time series a cluster can have.
        
        Return: 
        List of updated Dataset objects.
        """
        # merge clusters with <5 time series to the most similar cluster from the same data set
        updated_datasets = self._merge_small_clusters(datasets, min_nb_ts)

        # constrain clusters: each should have btw 5-15 ts
        updated_datasets = self._explode_large_clusters(datasets, max_nb_ts)

        return updated_datasets

    def are_cids_unique(self, datasets):
        """
        Checks if each cluster's ID is unique for the given list of data sets.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the clustered time series.
        
        Return: 
        List of Dataset objects
        """
        all_cids = []
        for cid in Dataset.yield_each_datasets_cluster_id(datasets):
            if cid in all_cids:
                return False
            all_cids.append(cid)
        return True

    def _get_dataset_ncc_scores(self, timeseries):
        """
        Measure the Normalized Cross-Correlation score over all pairs of time series in the data set.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series (each row is a time series)
        
        Return:
        List of NCC scores measured for all pairs of time series (e.g. dataset with 6 time series will produce a list of 15 NCC scores)
        """
        if timeseries.shape[0] > 1: # if there are >1 time series in the data set
            ncc_scores = []
            for i in range(1, len(timeseries.index)):
                for j in range(0, i):
                    values = _ncc_c(timeseries.iloc[i].values, timeseries.iloc[j].values)
                    ncc_scores.append(values[values.argmax()])

            return np.array(ncc_scores)
        else:
            return np.array([1.0])
    
    def _merge_small_clusters(self, datasets, min_nb_ts):
        """
        For each data set, merges the clusters having less than "min_nb_ts" time series to the most similar 
        cluster from the same data set.
        Updates the clusters assignment csv files.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        min_nb_ts -- minimum number of time series a cluster can have without requiring a merge.
        
        Return: 
        List of updated Dataset objects.
        """
        all_avg_ncc = {}

        # compute avg ncc score of each cluster
        for dataset, _, cluster, cluster_id in Dataset.yield_each_datasets_cluster(datasets):
            all_avg_ncc[f'{dataset.name}_{cluster_id}'] = np.mean(self._get_dataset_ncc_scores(cluster))

        # merging phase
        updated_datasets = []
        for dataset, timeseries, cluster, cluster_id in Dataset.yield_each_datasets_cluster(datasets):
            clusters_assignment = self.load_clusters(dataset.name)
            if cluster.shape[0] < min_nb_ts: # merge if not enough time series
                ncc_diffs = {}

                # compute average NCC score of each cluster
                for other_clusters_id in clusters_assignment['Cluster ID'].unique():
                    if cluster_id != other_clusters_id: # for all the other clusters from this data set
                        other_cluster = dataset.get_cluster_by_id(timeseries, other_clusters_id, clusters_assignment)

                        merged_cluster = pd.concat([other_cluster, cluster]) # merge the two clusters

                        if merged_cluster.shape[0] >= min_nb_ts:
                            # compute the avg ncc score difference (score with merge - score without merge)
                            other_cluster_avg_ncc = all_avg_ncc[f'{dataset.name}_{other_clusters_id}']
                            merged_cluster_avg_ncc = np.mean(self._get_dataset_ncc_scores(merged_cluster))
                            ncc_diffs[other_clusters_id] = merged_cluster_avg_ncc - other_cluster_avg_ncc

                # merge with the cluster that returned the largest positive diff
                ranked_ncc_diffs = sorted(ncc_diffs.items(), key=operator.itemgetter(1), reverse=True)
                selected_cluster_id = ranked_ncc_diffs[0][0]
                clusters_assignment['Cluster ID'].replace(cluster_id, selected_cluster_id, inplace=True)

                # update the avg ncc score of the merged cluster
                new_cluster = dataset.get_cluster_by_id(timeseries, selected_cluster_id, clusters_assignment)
                all_avg_ncc[f'{dataset.name}_{selected_cluster_id}'] = np.mean(self._get_dataset_ncc_scores(new_cluster))

            # save modified assignments to csv
            self.save_clusters(dataset, clusters_assignment)
            updated_datasets.append(dataset)
        return updated_datasets

    def _explode_large_clusters(self, datasets, max_nb_ts):
        """
        Explodes large clusters into multiple smaller ones.
        Updates the clusters assignment csv files.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        max_nb_ts -- maximum number of time series a cluster can have without requiring to be exploded.
        
        Return: 
        List of updated Dataset objects.
        """
        updated_datasets = []
        for dataset, timeseries, cluster, cluster_id in Dataset.yield_each_datasets_cluster(datasets):
            clusters_assignment = self.load_clusters(dataset.name)
            if cluster.shape[0] > max_nb_ts: # explode if too many time series

                # select all rows of this cluster
                rows = clusters_assignment.loc[clusters_assignment['Cluster ID'] == cluster_id]
                
                Ntotal = rows.shape[0]
                Nsections = int(np.ceil(Ntotal / max_nb_ts))
                # code taken from numpy "array_split" method
                # source (last consulted 09.11.2021): https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
                Neach_section, extras = divmod(Ntotal, Nsections)
                section_sizes = ([0] + extras * [Neach_section+1] + (Nsections-extras) * [Neach_section])
                div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

                # change the cid of each chunk of rows originally assigned to cluster_id's cluster
                for i in range(Nsections):
                    st = div_points[i]
                    end = div_points[i + 1]
                    clusters_assignment.at[rows[st:end].index.values, 'Cluster ID'] = cluster_id * 1000000 + i 
                
            # save modified assignments to csv
            self.save_clusters(dataset, clusters_assignment)
            updated_datasets.append(dataset)
        return updated_datasets