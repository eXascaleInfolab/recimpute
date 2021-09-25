"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
ClusterModel.py
@author: @chacungu
"""

from datetime import datetime
from functools import partial
import json
from kshape.core import kshape, zscore, _ncc_c
from multiprocessing import Pool
import numpy as np
import operator
import os
from os.path import normpath as normp
import pandas as pd
import time
from tqdm import tqdm

from Utils.Utils import Utils

class ClusterModel:
    """
    TODO description
    """

    CLUSTERING_RESULTS_DIR = './Clustering/Results/'
    GS_STATUS_FILE = normp(CLUSTERING_RESULTS_DIR + 'gridsearch_status.txt')
    GS_SCORES_FILE = normp(CLUSTERING_RESULTS_DIR + 'gridsearch_scores.json')
    CLUSTERING_STATUS_FILE = normp(CLUSTERING_RESULTS_DIR + 'clustering_status.txt')
    CONF = Utils.read_conf_file('clustering')

    # create necessary directories if not there yet
    if not os.path.exists(CLUSTERING_RESULTS_DIR):
        os.makedirs(CLUSTERING_RESULTS_DIR)

    def __init__(self):
        pass

    def _cluster_dataset(self, timeseries, nb_clusters, next_cid=0):
        """
        Clusters the time series from the given data set. The number of clusters can be specified.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series (each row is a time series)
        nb_clusters -- number of clusters to produce
        next_cid -- next cluster ID to assign (default 0)
        
        Return:
        DataFrame containing two columns (Time Series ID and Cluster ID) sorted by ascending order of column 1.
        """
        clusters = kshape(timeseries, nb_clusters)
        clusters_assignment = pd.DataFrame(data =
                                           [(tid, cid+next_cid) # time series id, assigned cluster's id
                                            for cid, (_, cluster_sequences) in enumerate(clusters)
                                            for tid in cluster_sequences], 
                                           columns=['Time Series ID', 'Cluster ID'])
        # sort by time series ID and not cluster ID
        clusters_assignment = clusters_assignment.sort_values(clusters_assignment.columns[0])
        return clusters_assignment

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
    
    def _clustering_reruns(self, dataset, nb_clusters, nb_reruns, accept_small_clusters=False, save_best_tocsv=False):
        """
        Calls multiple times the clustering method and scores each clustering assignment.
        
        Keyword arguments:
        dataset -- Dataset object containing the time series to cluster
        nb_clusters -- number of clusters to produce
        nb_reruns -- number of times clustering assignments must be produced and evaluated
        accept_small_clusters -- accepts all clusters even if they contain very few time series if True
        save_best_tocsv -- if True saves the best clustering assignments to csv (default False)
        
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
            clusters_assignment = self._cluster_dataset(timeseries, nb_clusters)

            if not accept_small_clusters:
                # if at least one cluster has less than 2 time series
                if any(v < 2 for v in clusters_assignment['Cluster ID'].value_counts().unique()):
                    if nb_retries < ClusterModel.CONF['GS_MAX_RETRIES']:
                        # dismiss & try to *redo* this run
                        print(f'TMP: %s\t #clusters %i, #run %i, #retry %i' % (dataset.name, nb_clusters, id_run, nb_retries))
                        nb_retries += 1
                        continue
                    else:
                        # dismiss & *skip* this run
                        nb_retries = 0
                        id_run += 1
                        reruns_scores.append(0)
                        continue
            nb_retries = 0

            # compute the score of each cluster
            allclusters_ncc_scores = []
            for cluster_id in clusters_assignment['Cluster ID'].unique(): # for each cluster
                # retrieve cluster
                clusters_ts = timeseries.loc[clusters_assignment['Cluster ID'] == cluster_id] 
                # measure normalized cross-correlation between each pair of time series of this cluster
                cluster_ncc_scores = self._get_dataset_ncc_scores(clusters_ts) if clusters_ts.shape[0] > 1 else [[1.0]]
                allclusters_ncc_scores.append(cluster_ncc_scores)
            # compute the run's score: sum( avg( ncc between each pair of time series in the cluster ) for each cluster )
            run_score = sum(np.mean(v) for v in allclusters_ncc_scores)
            # normalize the run's score by the number of clusters
            run_score /= len(clusters_assignment['Cluster ID'].unique())
            
            print(f'TMP: %s\t #clusters %i, #run %i, score %.3f' % (dataset.name, nb_clusters, id_run, run_score))

            # save the cluster assignment if it yields a better score than the previous runs
            if save_best_tocsv:
                if len(reruns_scores) < 1 or run_score > max(reruns_scores): # if first run or new score > current max
                    print(f'TMP: %s\t #clusters %i, #run %i, new best score %.3f' % (dataset.name, nb_clusters, id_run, run_score))
                    # save clusters assignments produced in this run as csv
                    dataset.save_cassignment(clusters_assignment)

            reruns_scores.append(run_score) # save the run's score

            id_run += 1
        return dataset, reruns_scores

    def _gridsearch(self, dataset):
        """
        Tries different number of clusters for a data set in order to decide which number is optimal.
        
        Keyword arguments:
        dataset -- Dataset object containing the time series to cluster
        
        Return:
        1. data set object containing the time series
        2. dict (with keys: 'min', 'max', 'avg', 'elapsed_times') containing results of the clustering for each tested number of clusters. 
        3. list containing the different number of clusters tested.
        """
        # load dataset
        timeseries = dataset.load_timeseries(transpose=True)
        gs_nb_runs = self._get_gs_nb_runs(timeseries)
        ds_scores = {'min': [], 'max': [], 'avg': [], 'elapsed_times': [], 'gs_nb_runs': gs_nb_runs}
        ds_nb_clusters_gridsearch = None

        # compute median normalized cross-correlation score between each pair of time series in the data set
        dataset_ncc_scores = None
        try:
            dataset_ncc_scores = self._get_dataset_ncc_scores(timeseries)
        except MemoryError:
            ds_scores = None
            ds_nb_clusters_gridsearch = None
            
        if dataset_ncc_scores is not None:
            # use clustering iff time series are not yet correlated enough
            if np.median(dataset_ncc_scores) < ClusterModel.CONF['NCC_MIN_THRESHOLD']:

                ds_nb_clusters_gridsearch = self._get_ds_gridsearch_range(timeseries, dataset.name)
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
                print(f'TMP: %s\t no clustering: %.3f' % (dataset.name, np.mean(dataset_ncc_scores)))
                ds_scores['avg'].append(np.mean(dataset_ncc_scores))
                ds_nb_clusters_gridsearch = None
                clusters_assignment = pd.DataFrame(list(zip(timeseries.index, np.zeros(len(timeseries.index), np.int8))),
                                                   columns=['Time Series ID', 'Cluster ID'])
                dataset.save_cassignment(clusters_assignment)
            
        return dataset, ds_scores, ds_nb_clusters_gridsearch

    def _final_clustering(self, dataset, nb_clusters_per_ds):
        """
        Clusters the data set's time series with the optimal number of clusters obtained previously through gridsearch.
        Saves the best clusters assignment to csv.
        
        Keyword arguments:
        dataset -- Dataset object containing the time series to cluster
        
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
                                                 self._get_clustering_nb_runs(timeseries), 
                                                 accept_small_clusters=True, save_best_tocsv=True)  # TODO
        else:
            clusters_assignment = pd.DataFrame(data = [(tid, 0) # time series id, assigned cluster's id
                                                        for tid in timeseries.index], 
                                               columns=['Time Series ID', 'Cluster ID'])
            clusters_assignment = clusters_assignment.sort_values(clusters_assignment.columns[0])
            dataset.save_cassignment(clusters_assignment)
        
        return dataset

    def _merge_small_clusters(self, datasets, min_nb_ts):
        """
        For each data set, merges the clusters having less than "min_nb_ts" time series to the most similar 
        cluster from the same data set.
        Updates the clusters assignment csv files.
        
        Keyword arguments:
        min_nb_ts -- minimum number of time series a cluster can have without requiring a merge.
        
        Return: -
        """
        all_avg_ncc = {}

        # compute avg ncc score of each cluster
        for dataset in tqdm(datasets): # for each data set
            timeseries = dataset.load_timeseries(transpose=True) # load data set's time series
            clusters_assignment = dataset.load_cassignment() # load clusters assignment
            for cluster_id in clusters_assignment['Cluster ID'].unique(): # for each cluster
                cluster = dataset.get_cluster_by_id(timeseries, cluster_id, clusters_assignment)
                all_avg_ncc[f'{dataset.name}_{cluster_id}'] = np.mean(self._get_dataset_ncc_scores(cluster))

        # merging phase
        for dataset in tqdm(datasets): # for each data set
            timeseries = dataset.load_timeseries(transpose=True) # load data set's time series
            clusters_assignment = dataset.load_cassignment() # load clusters assignment
            for cluster_id in clusters_assignment['Cluster ID'].unique(): # for each cluster
                cluster = dataset.get_cluster_by_id(timeseries, cluster_id, clusters_assignment)

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
            dataset.save_cassignment(clusters_assignment)

    def run(self, datasets):
        """
        For each data set, searches the optimal number of clusters to produce, performs multiple clustering tries
        to find the most accurate. Saves the clusters' assignment to a .csv file stored in the Dataset object.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series to cluster.
        
        Return: 
        List of Dataset objects
        """

        # ---
        # search for optimal number of clusters for each data set (gridsearch)
        nb_clusters_gridsearch = {}
        scores = {}
        datasets_not_processed = []

        with open(ClusterModel.GS_STATUS_FILE, 'w') as f:
            f.write('Gridsearch of following data sets started at %s:\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            for ds in datasets:
                f.write('- %s\n' % ds.name)
            f.write('\n')

        exception = None
        updated_datasets = []
        try:
            p = Pool() # create pool of threads (uses all available cores by default)
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
        
        with open(ClusterModel.GS_STATUS_FILE, 'a') as f:
            if exception is not None:
                f.write('Got exception %s.\n\n' % exception)
            if len(datasets_not_processed) > 0:
                f.write('Not enough memory to process:\n')
                for dataset_name in datasets_not_processed:
                    f.write('- %s\n' % dataset_name)
            f.write('Gridsearch ended at %s.\n\n\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            if exception is not None:
                raise e

        
        # ---
        # load results from disk if the results are not in memory
        try:
            scores
            nb_clusters_gridsearch
        except NameError: # variable is not in memory
            try:
                # load scores
                with open(ClusterModel.GS_SCORES_FILE, 'r') as f:
                    scores = json.load(f)
                    
                # load ranges
                nb_clusters_gridsearch = {}
                for dataset in datasets:
                    # load dataset
                    timeseries = dataset.load_timeseries(transpose=True)
                    # get range
                    nb_clusters_gridsearch[dataset.name] = self._get_ds_gridsearch_range(timeseries, dataset.name)
            except FileNotFoundError:
                raise FileNotFoundError('No clustering gridsearch results found on disk.')

        nb_clusters_per_ds = {}


        # ---
        # define optimal number of clusters for each dataset
        for dataset_name, measures in scores.items():
            if len(measures['avg']) <= 1:
                nb_clusters_per_ds[dataset_name] = 1
            else:
                x_maxima_sc = nb_clusters_gridsearch[dataset_name][np.argmax(measures['avg'], axis=0)]
                nb_clusters_per_ds[dataset_name] = x_maxima_sc


        # ---
        # apply "final" clustering: cluster each dataset N times with optimal number of clusters discovered by gridsearch 
        # and keep the assignment yielding the highest score    

        with open(ClusterModel.CLUSTERING_STATUS_FILE, 'w') as f:
            f.write('Clustering of following data sets started at %s:\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            for dataset in datasets:
                f.write('- %s\n' % dataset.name)
            f.write('\n')
            
        updated_datasets = []
        func = partial(self._final_clustering, nb_clusters_per_ds=nb_clusters_per_ds)
        with Pool() as p:
            for dataset in p.map(func, datasets):
                updated_datasets.append(dataset)
        datasets = updated_datasets
            
        with open(ClusterModel.CLUSTERING_STATUS_FILE, 'a') as f:
            f.write('Clustering ended at %s.\n\n\n' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


        # ---
        # merge clusters with <5 time series to the most similar cluster from the same data set
        self._merge_small_clusters(datasets, min_nb_ts=ClusterModel.CONF['MIN_NB_TS_PER_CLUSTER'])


        # ---
        # change all clusters' label (for all datasets) such that there are no dupplicates
        next_global_cid = 0
        for dataset in datasets:
            # load clusters assignment of each dataset
            clusters_assignment = dataset.load_cassignment()
            # update cluster ids such that there are no duplicates over all datasets
            for i, cluster_id in enumerate(clusters_assignment['Cluster ID'].unique()):
                clusters_assignment['Cluster ID'].replace(cluster_id, '#' + str(i + next_global_cid), inplace=True)
            clusters_assignment['Cluster ID'] = clusters_assignment['Cluster ID'].map(lambda v: int(v.replace('#', '')))
            next_global_cid += len(clusters_assignment['Cluster ID'].unique())
            # save modified assignments to csv
            dataset.save_cassignment(clusters_assignment)
        
        return datasets

    def _get_ds_space_complexity(self, timeseries):
        nb_ts, nb_features = timeseries.shape
        return nb_ts * nb_features

    def _get_ds_gridsearch_range(self, timeseries, dataset_name):
        if dataset_name in ClusterModel.CONF['GS_OPTIONAL_RANGES_HINTS']:
            _range = ClusterModel.CONF['GS_OPTIONAL_RANGES_HINTS'][dataset_name]
            return list(np.arange(_range['min'], _range['max']+1, _range['step'], dtype=int))
        else:
            complexity = self._get_ds_space_complexity(timeseries) # get complexity score of dataset
            # retrieve max number of tests (= max nb clusters to test) depending on the dataset's complexity
            for max_complexity, nb_tests in ClusterModel.CONF['GS_MAX_TESTS'].items():
                if complexity < max_complexity:
                    max_tests = nb_tests
                    break

            # maximum number of clusters that we could find for a dataset is #timeseries / 2
            max_nb_clusters = timeseries.shape[0] / 2

            # find and return the range of values to test
            step = np.ceil(max_nb_clusters / max_tests)
            range_ = list(np.arange(2, max_nb_clusters+1, step, dtype=int))
            return range_
        
    def _get_gs_scores_filename(self):
        if os.path.exists(ClusterModel.GS_SCORES_FILE):
            i = 1
            while True:
                new_filename = '.'.join(ClusterModel.GS_SCORES_FILE.split('.')[:-1]) + f' ({i}).' + ClusterModel.GS_SCORES_FILE.split('.')[-1]
                if not os.path.exists(new_filename):
                    os.rename(ClusterModel.GS_SCORES_FILE, new_filename)
                    break
                i += 1
        return ClusterModel.GS_SCORES_FILE
        
    def _get_nb_runs(self, timeseries, nb_runs_dict):
        if self._get_ds_space_complexity(timeseries) > 50000:
            if timeseries.shape[1] > 5000:
                return nb_runs_dict['large']
            print('medium')
            return nb_runs_dict['medium']
        print('small')
        return nb_runs_dict['small']
        
    def _get_gs_nb_runs(self, timeseries):
        return self._get_nb_runs(timeseries, ClusterModel.CONF['GS_NB_RUNS'])

    def _get_clustering_nb_runs(self, timeseries):
        return self._get_nb_runs(timeseries, ClusterModel.CONF['CLUSTERING_NB_RUNS'])
