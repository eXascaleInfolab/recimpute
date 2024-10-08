"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
ShapeBasedClustering.py
@author: @chacungu
"""

from datetime import datetime
from functools import partial
import json
from kshape.core import kshape as kshape_cpu
from kshape.core_gpu import kshape as kshape_gpu
import math
from multiprocessing import Pool
import numpy as np
import os
from os.path import normpath as normp
import pandas as pd
import time
import torch
from tqdm import tqdm

from Clustering.AbstractClustering import AbstractClustering
from Clustering.ConFree_kClustering import cluster as cfkc_cluster
from Datasets.Dataset import Dataset
from Utils.Utils import Utils

class ShapeBasedClustering(AbstractClustering):
    """
    Shape-based clustering model used to optimally cluster time series stored in Dataset objects. Uses k-Shape.
    """

    GS_SCORES_FILE = normp(AbstractClustering.CLUSTERS_DIR + 'sbc_gridsearch_scores.json')
    CLUSTERING_STATUS_FILE = normp(AbstractClustering.CLUSTERS_DIR + 'sbc_clustering_status.txt')
    CLUSTERS_FILENAMES_ID = '_sbc'
    CONF = Utils.read_conf_file('clustering')


    # constructor

    def __init__(self):
        pass


    # public methods

    def cluster_all_datasets(self, datasets):
        """
        Parallel - For each data set, searches the optimal number of clusters to produce, performs multiple clustering tries
        to find the most accurate. Saves the clusters' assignment to a .csv file stored in the Dataset object.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        
        Return: 
        List of Dataset objects
        """
        print('Clustering of following data sets started at %s:\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        for dataset in datasets:
            print('- %s' % dataset.name)
        print('\n')
            
        updated_datasets = []
        with Pool() as p:
            with tqdm(total=len(datasets)) as pbar:
                for updated_dataset in p.imap(self.cluster, datasets):
                    updated_datasets.append(updated_dataset)
                    pbar.update()
            
        print('Clustering ended at %s.\n\n\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        # change all clusters' ID (for all datasets) such that there are no duplicates
        updated_datasets = self.make_cids_unique(updated_datasets)
        return updated_datasets

    def cluster_all_datasets_seq(self, datasets):
        """
        Sequential - For each data set, searches the optimal number of clusters to produce, performs multiple clustering tries
        to find the most accurate. Saves the clusters' assignment to a .csv file stored in the Dataset object.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        
        Return: 
        List of Dataset objects
        """
        print('Clustering of following data sets started at %s:\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        for dataset in datasets:
            print('- %s' % dataset.name)
        print('\n')
            
        updated_datasets = []
        for dataset in tqdm(datasets, total=len(datasets)):
            try:
                updated_dataset = self.cluster(dataset)
                updated_datasets.append(updated_dataset)
            except MemoryError as e:
                print('%s got a MemoryError exception: %s' % (dataset.name, e))
            
            
        print('Clustering ended at %s.\n\n\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        # change all clusters' ID (for all datasets) such that there are no duplicates
        updated_datasets = self.make_cids_unique(updated_datasets)
        return updated_datasets

    def cluster(self, dataset):
        """
        Clusters the given data set's time series. Saves the clusters' assignment to a .csv file stored in the Dataset object.
        
        Keyword arguments:
        dataset -- Dataset objects containing the time series to cluster.
        
        Return: 
        Updated Dataset object
        """
        timeseries = dataset.load_timeseries(transpose=True)

        # add 2 to the correlation score to avoid dividing by 0 in some computations
        corr_offset = 2 # changing this impacts the moving_thresh param which may then not be optimal !

        labels = cfkc_cluster(
            self.kshape_helper, 
            lambda timeseries: self._get_dataset_mean_corr(timeseries) +corr_offset, 
            timeseries, 
            obj_thresh = ShapeBasedClustering.CONF['CLUSTER_ACCEPTANCE_THRESHOLD'] +corr_offset, 
            init_obj_thresh = ShapeBasedClustering.CONF['INIT_ACCEPTANCE_THRESHOLD'] +corr_offset, 
            sim_cluster_thresh = ShapeBasedClustering.CONF['SIMILAR_CLUSTER_THRESHOLD'] +corr_offset, 
            centroid_dist_thresh = ShapeBasedClustering.CONF['CENTROID_DIST_THRESHOLD'] +corr_offset,
            k_perc = ShapeBasedClustering.CONF['TS_PERC_TO_COMPUTE_K'], 
            security_limit = 5, 
            max_iter = ShapeBasedClustering.CONF['MAX_ITER'], 
            id = dataset.name,
            apply_merging = ShapeBasedClustering.CONF['APPLY_MERGING']
        )

        # create the clusters' assignments data frame
        data = [
            (tid, cid) # time series id, assigned cluster's id
            for tid, cid in zip(timeseries.index, labels)
        ]
        clusters_assignment = pd.DataFrame(data=data, columns=['Time Series ID', 'Cluster ID']).sort_values('Time Series ID')
        assert sorted(timeseries.index) == sorted(clusters_assignment['Time Series ID'].tolist())
        self.save_clusters(dataset, clusters_assignment)
        return dataset

    def kshape_helper(self, k, X):
        """
        Clusters the given time series using the k-Shape algorithm with an objective of k clusters.
        It returns the index of the cluster each sample belongs to.
        
        Keyword arguments:
        k -- number of clusters k-Shape will try producing
        X -- pandas DataFrame of time series to cluster (each row is one time series)
        
        Return: 
        List of labels: index of the cluster each sample belongs to.
        """
        clusters_assignment = self._cluster_timeseries(X, k)
        return clusters_assignment['Cluster ID'].to_numpy()

    def save_clusters(self, dataset, cassignment):
        """
        Saves the given clusters to CSV.
        
        Keyword arguments: 
        dataset -- Dataset object to which the clusters belong
        cassignment -- Pandas DataFrame containing clusters' assignment of the data set's time series. 
                       Its index is the same as the real world data set of this object. The associated 
                       values are the clusters' id to which are assigned the time series.
        
        Return: -
        """
        cassignment_filename = self._get_cassignment_filename(dataset.name)
        dataset.cids = cassignment['Cluster ID'].unique().tolist()
        cassignment.to_csv(cassignment_filename, index=False)

    def load_clusters(self, dataset_name):
        """
        Loads the clusters of the given data set.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the clusters belong
        
        Return: 
        Pandas DataFrame containing clusters' assignment of the data set's time series. Its index is the same 
        as the real world data set of this object. The associated values are the clusters' id to which are
        assigned the time series. Two columns: Time Series ID, Cluster ID.
        """
        cassignment_filename = self._get_cassignment_filename(dataset_name)
        clusters_assignment = pd.read_csv(cassignment_filename)
        return clusters_assignment

    
    # private methods

    def _cluster_timeseries(self, timeseries, nb_clusters):
        """
        Clusters the given time series.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series (each row is a time series)
        nb_clusters -- number of clusters to produce
        
        Return:
        DataFrame containing two columns (Time Series ID and Cluster ID) sorted by ascending order of column 1.
        """
        kshape = kshape_gpu if torch.cuda.is_available() else kshape_cpu
        clusters = kshape(np.expand_dims(timeseries.to_numpy(), axis=2), nb_clusters)
        clusters_assignment = pd.DataFrame(data =
                                           [(tid, cid) # time series id, assigned cluster's id
                                            for cid, (_, cluster_sequences) in enumerate(clusters)
                                            for tid in cluster_sequences], 
                                           columns=['Time Series ID', 'Cluster ID'])
        # sort by time series ID and not cluster ID
        clusters_assignment = clusters_assignment.sort_values(clusters_assignment.columns[0])
        return clusters_assignment

    def _get_cassignment_filename(self, dataset_name):
        """
        Returns the filename of the clusters for the given data set's name.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the clusters belong
        
        Return: 
        Filename of the clusters for the given data set's name.
        """
        return normp(
            AbstractClustering.CLUSTERS_DIR + \
            f'/{dataset_name}{ShapeBasedClustering.CLUSTERS_FILENAMES_ID}{AbstractClustering.CLUSTERS_APPENDIX}')

    def _compute_run_score(self, timeseries, cassignment):
        """
        Computes a clustering run's score.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series (each row is a time series)
        cassignment -- Pandas DataFrame containing clusters' assignment of the data set's time series. 
                       Its index is the same as the real world data set of this object. The associated 
                       values are the clusters' id to which are assigned the time series.
        
        Return:
        Run score
        """
        # compute the score of each cluster
        allclusters_mean_ncc_scores = []
        for cluster_id in cassignment['Cluster ID'].unique(): # for each cluster
            # retrieve cluster
            clusters_ts = timeseries.loc[cassignment['Cluster ID'] == cluster_id] 
            # measure normalized cross-correlation between each pair of time series of this cluster
            cluster_mean_ncc_score = self._get_dataset_mean_ncc_score(clusters_ts)
            allclusters_mean_ncc_scores.append(cluster_mean_ncc_score)
        # compute the run's score: sum( avg( ncc between each pair of time series in the cluster ) for each cluster )
        run_score = sum(allclusters_mean_ncc_scores)
        # normalize the run's score by the number of clusters
        run_score /= len(cassignment['Cluster ID'].unique())
        return run_score

    

    # to delete eventually (still used in the clusterings comparison experiment to compare against this gridsearch approach)

    def cluster_old(self, dataset):
        """
        Clusters the given data set's time series. Saves the clusters' assignment to a .csv file stored in the Dataset object.
        
        Keyword arguments:
        dataset -- Dataset objects containing the time series to cluster.
        
        Return: 
        Updated Dataset object
        """
        timeseries = dataset.load_timeseries(transpose=True)

        # cluster iteratively the time series using k-Shape
        clusters = self._cluster_subroutine(timeseries, dataset.name)

        # merge clusters if correlation remains high
        merged_clusters = self._merge_clusters(clusters)

        # create the clusters' assignments data frame
        data = [
            (tid, cid) # time series id, assigned cluster's id
            for cid, cluster in enumerate(merged_clusters)
            for tid in cluster.index
        ]
        clusters_assignment = pd.DataFrame(data=data, columns=['Time Series ID', 'Cluster ID']).sort_values('Time Series ID')
        assert sorted(timeseries.index) == sorted(clusters_assignment['Time Series ID'].tolist())
        self.save_clusters(dataset, clusters_assignment)
        return dataset

    def _cluster_subroutine(self, original_timeseries, ds_name, security_limit=5, max_iter=7500):
        """
        Clusters the given data set's time series.
        
        Keyword arguments:
        original_timeseries -- Pandas DataFrame containing the data set's time series (each row is a time series)
        ds_name -- name of the data set being clustered
        security_limit -- number of iterations without new valid cluster before "helping" the clustering algorithm (default: 5)
        max_iter -- maximum number of iterations before a force-stop (default: 10000)
        
        Return: 
        List of resulting clusters (each cluster is a Pandas DataFrame of time series)
        """
        stack = [original_timeseries] # contains the clusters that have not been handled yet
        result = [] # contains the clusters that have been accepted
        security_count = 0
        iter_count = 0
        while len(stack) > 0:

            timeseries = stack.pop()
            assert timeseries.shape[0] > 0
    
            threshold = ShapeBasedClustering.CONF['CLUSTER_ACCEPTANCE_THRESHOLD'] if iter_count > 0 \
                        else ShapeBasedClustering.CONF['INIT_ACCEPTANCE_THRESHOLD']
            if timeseries.shape[0] == 1 or self._get_dataset_mean_corr(timeseries) >= threshold:#self._get_dataset_mean_ncc_score(timeseries) >= threshold:
                security_count = 0
                # correlation is high enough: accept the cluster
                result.append(timeseries)
            elif timeseries.shape[0] == 2:
                security_count = 0
                # accept the two remaining time series as individual clusters
                result.append(timeseries.iloc[0].to_frame().T)
                result.append(timeseries.iloc[1].to_frame().T)
            else:
                # correlation in this cluster is not high enough: cluster with k-Shape
                security_count += 1

                K = max(2, math.floor(timeseries.shape[0] * ShapeBasedClustering.CONF['TS_PERC_TO_COMPUTE_K']))
                if security_count >= 15 * security_limit: 
                    # even with help k-Shape couldn't create valid clusters for too many iterations
                    # manually split the current cluster in 2 hoping it unstucks the algorithms
                    stack.insert(0, timeseries.iloc[0::2])
                    stack.insert(0, timeseries.iloc[1::2])
                else:
                    if security_count >= security_limit: 
                        # k-Shape is having difficulties creating clusters that match the criteria
                        # to unstuck it, increase K to help create valid clusters
                        K += (security_count // (security_limit-1)) * K
                        K = min(timeseries.shape[0], K)
                    
                    clusters_assignment = self._cluster_timeseries(timeseries, K)
                    clusters_assignment['Time Series ID'] = timeseries.index
                    
                    for cluster_id in clusters_assignment['Cluster ID'].unique():
                        ts_ids_for_cid = clusters_assignment['Time Series ID'][clusters_assignment['Cluster ID'] == cluster_id]
                        clusters_timeseries = timeseries.loc[ts_ids_for_cid]
                        stack.insert(0, clusters_timeseries)
            iter_count += 1
            if iter_count >= max_iter:
                print('Max iteration reached for %s! %i clusters did not meet all requirements yet.' % (ds_name, len(stack)))
                result.extend(stack)
                break
        print('clustering subroutine done for %s w/ %i iterations' % (ds_name, iter_count))
        return result

    def _merge_clusters(self, clusters):
        """
        Tries to merge clusters if the correlation remains high.
        
        Keyword arguments:
        clusters -- list of resulting clusters (each cluster is a Pandas DataFrame of time series)
        
        Return:
        Updated clusters: list of updated clusters (each cluster is a Pandas DataFrame of time series)
        """
        clusters = dict(zip(range(0, len(clusters)), clusters))
        
        # init: compute a centroid for each cluster as well as the avg correlation inside each cluster
        all_centroids, all_corrs = {}, {}
        for cid, cluster_ts in clusters.items():
            all_centroids[cid] = cluster_ts.mean()
            all_corrs[cid] = self._get_dataset_mean_corr(cluster_ts)

        # for each cluster
        for cid in list(clusters.keys()):
            # retrieve the cluster's time series and centroid
            cluster_ts = clusters[cid]
            centroid = all_centroids[cid]

            # compute the correlation btw the cluster's centroid and all other clusters' centroid
            all_centroid_corrs = [(other_cid, centroid.corr(all_centroids[other_cid])) for other_cid, _ in clusters.items()]
            
            # identify a list of similar clusters
            similar_clusters = filter(lambda item: item[1] >= ShapeBasedClustering.CONF['SIMILAR_CLUSTER_THRESHOLD'] and item[0] != cid, 
                                      all_centroid_corrs)

            best_corr, best_corr_diff, best_cid = -np.inf, -np.inf, None
            # for each similar cluster
            for other_cid, _ in similar_clusters:
                other_cluster_ts = clusters[other_cid]
                # compute the correlation of the cluster's sequences added to those of the "similar" cluster
                new_corr = self._get_dataset_mean_corr(pd.concat([cluster_ts, other_cluster_ts]))
                new_corr_diff = new_corr - all_corrs[other_cid]
                corr_perc = new_corr / all_corrs[other_cid]
                
                clusters_merging_threshold = ShapeBasedClustering.CONF['CLUSTERS_MERGING_THRESHOLD'] if other_cluster_ts.shape[0] > 1 \
                                             else ShapeBasedClustering.CONF['CLUSTERS_MERGING_THRESHOLD_SINGLE_TS']

                # is this similar cluster the best candidate for merging
                if new_corr_diff > best_corr_diff and corr_perc >= clusters_merging_threshold:
                    best_corr = new_corr
                    best_corr_diff = new_corr_diff
                    best_cid = other_cid
                    
            if best_cid != None:
                # merge cid and best_cid
                merged_cluster_ts = pd.concat([cluster_ts, clusters[best_cid]])
                clusters[best_cid] = merged_cluster_ts
                all_centroids[best_cid] = merged_cluster_ts.mean()
                all_corrs[best_cid] = self._get_dataset_mean_corr(merged_cluster_ts)
                
                del clusters[cid]
                del all_centroids[cid]
                del all_corrs[cid]
                
        return list(clusters.values())

    def old__cluster_all_datasets(self, datasets):
        """
        For each data set, searches the optimal number of clusters to produce, performs multiple clustering tries
        to find the most accurate. Saves the clusters' assignment to a .csv file stored in the Dataset object.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        
        Return: 
        List of Dataset objects
        """
        # search for optimal number of clusters for each data set (gridsearch)
        scores, nb_clusters_gridsearch = self._run_gridsearch(datasets)

        # define optimal number of clusters for each dataset
        nb_clusters_per_ds = {}
        for dataset_name, measures in scores.items():
            if len(measures['avg']) <= 1:
                nb_clusters_per_ds[dataset_name] = 1
            else:
                x_maxima_sc = nb_clusters_gridsearch[dataset_name][np.argmax(measures['avg'], axis=0)]
                nb_clusters_per_ds[dataset_name] = x_maxima_sc

        # apply "final" clustering: cluster each dataset N times with optimal number of clusters discovered by gridsearch 
        # and keep the assignment yielding the highest score    
        datasets = self._run_final_clustering(datasets, nb_clusters_per_ds)

        # merge clusters with <5 time series to the most similar cluster from the same data set
        # and divide large clusters into smaller ones such that each has between 5 and 15 time series
        # datasets = self.apply_constraints(datasets, 
        #                                   min_nb_ts=self.CONF['MIN_NB_TS_PER_CLUSTER'],
        #                                   max_nb_ts=self.CONF['MAX_NB_TS_PER_CLUSTER'])

        # change all clusters' ID (for all datasets) such that there are no duplicates
        datasets = self.make_cids_unique(datasets)
        return datasets

    def old__cluster(self, dataset):
        """
        Searches the optimal number of clusters to produce for this data set, performs multiple clustering tries
        to find the most accurate. Saves the clusters' assignment to a .csv file stored in the Dataset object.
        
        Keyword arguments:
        dataset -- Dataset objects containing the time series to cluster.
        
        Return: 
        Updated Dataset object
        """
        # search for optimal number of clusters for the data set (gridsearch)
        dataset, res_scores, res_range = self._gridsearch(dataset)

        # define optimal number of clusters for each dataset
        if len(res_scores['avg']) <= 1:
            nb_clusters = 1
        else:
            x_maxima_sc = res_range[np.argmax(res_scores['avg'], axis=0)]
            nb_clusters = x_maxima_sc

        # apply "final" clustering: cluster each dataset N times with optimal number of clusters discovered by gridsearch 
        # and keep the assignment yielding the highest score    
        dataset = self._final_clustering(dataset, {dataset.name: nb_clusters})

        return dataset

    def _check_clusters_validity(self, cassignment, accept_mono_ts_clusters):
        """
        Checks if the produced clusters are valid (at least 2 sequence per cluster).
        
        Keyword arguments:
        cassignment -- Pandas DataFrame containing clusters' assignment of the data set's time series. 
                       Its index is the same as the real world data set of this object. The associated 
                       values are the clusters' id to which are assigned the time series.
        accept_mono_ts_clusters -- accepts all clusters even if they contain only 1 sequence if True
        
        Return:
        True if the produced clusters are valid, False otherwise.
        """
        return accept_mono_ts_clusters or all(v >= 2 for v in cassignment['Cluster ID'].value_counts())

    def _invalidate_clustering_run(self, dataset, nb_clusters, id_run, nb_retries, reruns_scores):
        """
        Invalidates a clustering run (produced clusters were invalid).
        
        Keyword arguments:
        dataset -- Dataset object containing the time series to cluster
        nb_clusters -- number of clusters to produce
        id_run -- id of the current clustering run
        nb_retries -- current number of retries for this number of clusters to produce
        reruns_scores -- list of scores for each run
        
        Return:
        1. id of the current clustering run (updated)
        2. current number of retries for this number of clusters to produce (updated)
        3. list of scores for each run (updated)
        """
        if nb_retries < ShapeBasedClustering.CONF['GS_MAX_RETRIES']:
            # dismiss this clustering try & *redo* this run
            print(f'TMP: %s\t #clusters %i, #run %i, #retry %i' % (dataset.name, nb_clusters, id_run, nb_retries))
            nb_retries += 1
        else:
            # dismiss this clustering try & *skip* this run
            nb_retries = 0
            id_run += 1
            reruns_scores.append(0)
        return id_run, nb_retries, reruns_scores

    def _clustering_reruns(self, dataset, nb_clusters, nb_reruns, accept_small_clusters=True, save_best_tocsv=False):
        """
        Calls multiple times the clustering method on the Dataset object's time series (for a given number of clusters 
        to produce) and scores each clustering assignment.
        
        Keyword arguments:
        dataset -- Dataset object containing the time series to cluster
        nb_clusters -- number of clusters to produce
        nb_reruns -- number of times clustering assignments must be produced and evaluated for this number of clusters
        accept_small_clusters -- accepts all clusters even if they contain very few time series if True
        save_best_tocsv -- if True saves the best clustering assignments to csv (default True)
        
        Return:
        1. Dataset object containing the time series.
        2. List of scores measured after each run.
        """
        reruns_scores = [] # stores the scores of each run
        id_run = 0
        nb_retries = 0
        timeseries = dataset.load_timeseries(transpose=True)

        while id_run < nb_reruns:
            # cluster dataset
            clusters_assignment = self._cluster_timeseries(timeseries, nb_clusters)

            # check clusters validity
            if not self._check_clusters_validity(clusters_assignment, accept_small_clusters):
                id_run, nb_retries, reruns_scores = self._invalidate_clustering_run(dataset, nb_clusters, 
                                                                                    id_run, nb_retries, reruns_scores)
                continue

            # produced clusters match requirements
            nb_retries = 0

            # compute the clustering run's score
            run_score = self._compute_run_score(timeseries, clusters_assignment)
            print(f'TMP: %s\t #clusters %i, #run %i, score %.3f' % (dataset.name, nb_clusters, id_run, run_score))

            # save the cluster assignment if it yields a better score than the previous runs
            if save_best_tocsv:
                if len(reruns_scores) < 1 or run_score > max(reruns_scores): # if first run or new score > current max
                    print(f'TMP: %s\t #clusters %i, #run %i, new best score %.3f' % (dataset.name, nb_clusters, id_run, run_score))
                    # save clusters assignments produced in this run as csv
                    self.save_clusters(dataset, clusters_assignment)

            reruns_scores.append(run_score) # save the run's score

            id_run += 1
        return dataset, reruns_scores

    def _gridsearch(self, dataset):
        """
        Tries different number of clusters for a data set in order to decide which number is optimal.
        
        Keyword arguments:
        dataset -- Dataset object containing the time series to cluster
        
        Return:
        1. Dataset object containing the time series
        2. dict (with keys: 'min', 'max', 'avg', 'elapsed_times') containing results of the clustering for each tested number of clusters. 
        3. list containing the different number of clusters tested.
        """
        # load dataset
        timeseries = dataset.load_timeseries(transpose=True)
        gs_nb_runs = self._get_nb_runs(dataset, ShapeBasedClustering.CONF['GS_NB_RUNS'])
        ds_scores = {'min': [], 'max': [], 'avg': [], 'elapsed_times': [], 'gs_nb_runs': gs_nb_runs}
        ds_nb_clusters_gridsearch = None

        # compute median normalized cross-correlation score between each pair of time series in the data set
        dataset_mean_ncc_score = None
        try:
            dataset_mean_ncc_score = self._get_dataset_mean_ncc_score(timeseries)
        except MemoryError:
            ds_scores = None
            ds_nb_clusters_gridsearch = None
            
        if dataset_mean_ncc_score is not None:
            # use clustering iff time series are not yet correlated enough
            if dataset_mean_ncc_score < ShapeBasedClustering.CONF['NCC_MIN_THRESHOLD']:


                ds_nb_clusters_gridsearch = self._get_ds_gridsearch_range(dataset)
                print('TMP: ', dataset.name, '\n\trange:', ds_nb_clusters_gridsearch)

                try:
                    for nb_clusters in ds_nb_clusters_gridsearch:
                        start_time = time.time()

                        # multiple runs of clustering for each number of clusters -> final scores are the average
                        dataset, reruns_scores = self._clustering_reruns(dataset, nb_clusters, gs_nb_runs)

                        elapsed_time = time.time() - start_time

                        # save scores of the runs done with this number of clusters
                        ds_scores['min'].append(min(reruns_scores))
                        ds_scores['max'].append(max(reruns_scores))
                        ds_scores['avg'].append(np.mean(reruns_scores)) # average of rerun scores
                        ds_scores['elapsed_times'].append(elapsed_time)
                except MemoryError:
                    ds_scores = None
                    ds_nb_clusters_gridsearch = None
            else:
                # no clustering needed
                print(f'TMP: %s\t no clustering: %.3f' % (dataset.name, dataset_mean_ncc_score))
                ds_scores['avg'].append(dataset_mean_ncc_score)
                ds_nb_clusters_gridsearch = None
                clusters_assignment = pd.DataFrame(list(zip(timeseries.index, np.zeros(len(timeseries.index), np.int8))),
                                                   columns=['Time Series ID', 'Cluster ID'])
                self.save_clusters(dataset, clusters_assignment)
            
        return dataset, ds_scores, ds_nb_clusters_gridsearch

    def _run_gridsearch(self, datasets):
        """
        Runs the gridsearch step on each data set's time series.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        
        Return:
        1. dict with data sets' name as keys and, as values, a dict (with keys: 'min', 'max', 'avg', 'elapsed_times') 
           containing results of the clustering for each tested number of clusters
        2. dict with data sets' name as keys and, as values, a list containing the different number of clusters tested
        """
        nb_clusters_gridsearch = {}
        scores = {}
        datasets_not_processed = []

        print('Clustering gridsearch of following data sets started at %s:\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        for ds in datasets:
            print('- %s' % ds.name)
        print('\n')

        exception = None
        updated_datasets = []
        try:
            nb_workers = ShapeBasedClustering.CONF['GS_NB_WORKERS']
            p = Pool(processes= nb_workers if nb_workers > 0 else None) # create pool of threads
            with open(self._get_gs_scores_filename(), 'w') as f:
                f.write('{')
                for dataset, res_scores, res_range in p.imap_unordered(self._gridsearch, datasets):
                    updated_datasets.append(dataset)
                    if res_scores is not None:
                        scores[dataset.name] = res_scores    
                        nb_clusters_gridsearch[dataset.name] = res_range
                        f.write('"%s": %s, ' % (dataset.name, json.dumps(res_scores)))
                        print('TMP: %s\t\t done' % dataset.name)
                    else:
                        datasets_not_processed.append(dataset.name)
                        print('TMP: %s\t\t done: not enough memory.' % dataset.name)
                f.write('}')
            p.close()
        except KeyboardInterrupt as e:
            print('got ^C while pool mapping, terminating the pool')
            p.terminate()
            print('pool is terminated')
            raise e
        except Exception as e:
            print('got exception: %r, terminating the pool' % (e,))
            p.terminate()
            print('pool is terminated')
            exception = e
        finally:
            print('joining pool processes')
            p.join()
            print('join complete')
            datasets = updated_datasets
        
        if exception is not None: print('Got exception %s.\n\n' % exception)
        if len(datasets_not_processed) > 0:
            print('Not enough memory to process:\n')
            for dataset_name in datasets_not_processed:
                print('- %s\n' % dataset_name)
        print('Gridsearch ended at %s.\n\n\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        if exception is not None: raise exception

        return scores, nb_clusters_gridsearch

    def _load_gridsearch_results(self, datasets):
        """
        Loads the gridsearch results from disk.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        
        Return:
        1. dict with data sets' name as keys and, as values, a dict (with keys: 'min', 'max', 'avg', 'elapsed_times') 
           containing results of the clustering for each tested number of clusters
        2. dict with data sets' name as keys and, as values, a list containing the different number of clusters tested
        """
        try:
            # load scores
            with open(ShapeBasedClustering.GS_SCORES_FILE, 'r') as f:
                scores = json.load(f)
                
            # load ranges
            nb_clusters_gridsearch = {}
            for dataset in datasets:
                # get range
                nb_clusters_gridsearch[dataset.name] = self._get_ds_gridsearch_range(dataset)

            return scores, nb_clusters_gridsearch
            
        except FileNotFoundError:
            raise FileNotFoundError('No clustering gridsearch results found on disk.')

    def _final_clustering(self, dataset, nb_clusters_per_ds):
        """
        Clusters the data set's time series with the optimal number of clusters obtained previously with gridsearch.
        Saves the best clusters assignment to csv.
        
        Keyword arguments:
        dataset -- Dataset object containing the time series to cluster
        nb_clusters_per_ds -- dict with data sets' name as keys and their optimal number of clusters to produce as values
        
        Return: 
        Data set object containing the time series
        """
        # load dataset
        timeseries = dataset.load_timeseries(transpose=True)
        # retrieve optimal number of clusters for this dataset
        nb_clusters = nb_clusters_per_ds[dataset.name]
        if nb_clusters > 1:
            # cluster dataset    
            dataset, _ = self._clustering_reruns(dataset, nb_clusters, 
                                                 self._get_nb_runs(dataset, ShapeBasedClustering.CONF['CLUSTERING_NB_RUNS']), 
                                                 accept_small_clusters=True, save_best_tocsv=True)
        else:
            clusters_assignment = pd.DataFrame(data = [(tid, 0) # time series id, assigned cluster's id
                                                        for tid in timeseries.index], 
                                               columns=['Time Series ID', 'Cluster ID'])
            clusters_assignment = clusters_assignment.sort_values(clusters_assignment.columns[0])
            self.save_clusters(dataset, clusters_assignment)
        
        return dataset

    def _run_final_clustering(self, datasets, nb_clusters_per_ds):
        """
        Runs the final clustering step on each data set's time series.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        nb_clusters_per_ds -- dict with data sets' name as keys and their optimal number of clusters to produce as values
        
        Return:
        List of Dataset objects containing the time series.
        """
        print('Clustering of following data sets started at %s:\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        for dataset in datasets:
            print('- %s' % dataset.name)
        print('\n')
            
        updated_datasets = []
        func = partial(self._final_clustering, nb_clusters_per_ds=nb_clusters_per_ds)
        with Pool() as p:
            for dataset in p.map(func, datasets):
                updated_datasets.append(dataset)
            
        print('Clustering ended at %s.\n\n\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        return updated_datasets

    def _get_ds_gridsearch_range(self, dataset):
        """
        Returns the range of number of clusters to try during gridsearch for a given data set.
        
        Keyword arguments:
        dataset -- Dataset object containing the time series to cluster
        
        Return: 
        Range of number of clusters to try during gridsearch
        """
        if dataset.name in ShapeBasedClustering.CONF['GS_OPTIONAL_RANGES_HINTS']: 
            # if range of this data set is specified in the conf. file
            _range = ShapeBasedClustering.CONF['GS_OPTIONAL_RANGES_HINTS'][dataset.name]
            return list(np.arange(_range['min'], _range['max']+1, _range['step'], dtype=int))

        # else: compute the range dynamically
        complexity = dataset.get_space_complexity() # get complexity score of dataset
        # retrieve max number of tests (= max nb clusters to test) depending on the dataset's complexity
        for max_complexity, nb_tests in sorted(ShapeBasedClustering.CONF['GS_MAX_TESTS'].items()):
            if complexity < max_complexity:
                max_tests = nb_tests
                break

        # maximum number of clusters that we could find for a dataset is #timeseries / 2
        max_nb_clusters = dataset.nb_timeseries / 2

        # find and return the range of values to test
        step = np.ceil(max_nb_clusters / max_tests)
        range_ = list(np.arange(2, max_nb_clusters+1, step, dtype=int))
        return range_
        
    def _get_nb_runs(self, dataset, nb_runs_dict):
        """
        Returns the number of clustering runs to evaluate for a given data set and a number of clusters to produce.
        
        Keyword arguments:
        dataset -- Dataset object containing the time series to cluster
        nb_runs_dict -- dict containing the number of runs to perform depending on the data set's 
                        complexity ('small', 'medium', or 'large').
        
        Return: 
        Number of clustering runs to evaluate for a given data set and a number of clusters to produce
        """
        if dataset.get_space_complexity() > 50000:
            if dataset.timeseries_length > 5000:
                return nb_runs_dict['large']
            print('medium')
            return nb_runs_dict['medium']
        print('small')
        return nb_runs_dict['small']

    def _get_gs_scores_filename(self):
        """
        Returns filename of the gridsearch scores' file.
        
        Keyword arguments: -
        
        Return: 
        Filename of the gridsearch scores' file.
        """
        if os.path.exists(ShapeBasedClustering.GS_SCORES_FILE): 
            # if a gridsearch scores' file already exists
            i = 1
            while True:
                # rename existing files: increment a counter in their name
                new_filename = '.'.join(ShapeBasedClustering.GS_SCORES_FILE.split('.')[:-1]) + f' ({i}).' + ShapeBasedClustering.GS_SCORES_FILE.split('.')[-1]
                if not os.path.exists(new_filename):
                    os.rename(ShapeBasedClustering.GS_SCORES_FILE, new_filename)
                    break
                i += 1
        return ShapeBasedClustering.GS_SCORES_FILE

    