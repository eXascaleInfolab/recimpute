"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
ImputeBenchLabeler.py
@author: @chacungu
"""

import os
from os.path import normpath as normp
import pandas as pd
import shutil
import subprocess
import time
import warnings

from Labeling.AbstractLabeler import AbstractLabeler
from Utils.Utils import Utils

class ImputeBenchLabeler(AbstractLabeler):
    """
    TODO
    """

    LABELS_DIR = normp('./Labeling/ImputationTechniques/labels/')
    LABELS_APPENDIX = '_labels.csv'
    CONF = Utils.read_conf_file('imputebenchlabeler')

    # create necessary directories if not there yet
    Utils.create_dirs_if_not_exist([LABELS_DIR])


    # constructor

    def __init__(self):
        super().__init__()


    # public methods

    def label(self, datasets):
        # TODO
        updated_datasets = []
        for dataset in datasets:
            tmp_labels = []
            timeseries = dataset.load_timeseries(transpose=True)
            for cluster, cluster_id, _ in dataset.yield_all_clusters(timeseries):
                print('Running benchmark for cluster %i' % cluster_id)
                benchmark_results = self._label_cluster(timeseries, cluster, cluster_id)
                tmp_labels.append((cluster_id, benchmark_results))
            tmp_labels_df = pd.DataFrame(tmp_labels, columns=['Cluster ID', 'Benchmark Results'])
            
            dataset.set_labeler(self.__class__)
            dataset.save_labels(tmp_labels_df)
            updated_datasets.append(dataset)
        return updated_datasets


    # private methods

    def _label_cluster(self, all_timeseries, cluster, cluster_id):
        # TODO
        # select the sequences to give to the benchmark (either the whole cluster or the whole data set)
        sequences_to_use = cluster if not ImputeBenchLabeler.CONF['USE_BCHMK_ON_DATASET'] \
                           else pd.concat([cluster.iloc[0].to_frame(), all_timeseries])

        # call benchmark on the cluster's time series
        benchmark_results, _ = self._run_benchmark(sequences_to_use.T, 
                                                   algorithms='all',
                                                   scenario=ImputeBenchLabeler.CONF['BENCHMARK_SCENARIO'], 
                                                   errors=ImputeBenchLabeler.CONF['BENCHMARK_ERRORS'],
                                                   id=cluster_id,
                                                   plots=False,
                                                   delete_results=True)

        return benchmark_results.to_dict()

    def _run_benchmark(self, timeseries, algorithms, scenario, errors, id, plots=False, delete_results=True):
        """
        Runs the ImputeBench benchmark on given time series with specified algorithms and scenario. 
        If on Windows, wsl must be installed.
        
        Keyword arguments:
        timeseries -- DataFrame containing the time series (each column is a time series)
        algorithms -- string of the algorithm's name to run (or "all" to run all algorithms) or list of strings
                      to specify multiple algorithms' names to run
        scenario -- string of the scenario's name to run - ONLY "miss_perc" can be used as of now
        errors -- list of string for the errors' names to get in the returned DataFrame
        id -- custom id to incorporate in the temporary files/folders names
        plots -- creates reconstruction plots if True (default True)
        delete_results -- deletes the results folder if True (default True)
        
        Return:
        1. DataFrame containing the different scores. Multi index: "algorithms" & "scenario".
        2. Temporary file/folders name used for the benchmark
        """    
        # transform arguments to command line compatible arguments
        if isinstance(algorithms, list):
            algorithms = ','.join(algorithms)
            
        if ',' in scenario:
            raise Exception('Only one scenario can be used at a time.')
        if scenario != 'miss_perc':
            raise Exception('Only "miss_perc" scenario can be used as of now')
        if timeseries.shape[0] <= timeseries.shape[1]: # #features <= #time series
            warnings.warn('Warning: data set has more time series than features. Please double-check the time series\' format: each column should be one time series.')
            
        dataset_name = ('tmp_%s_%s' % (id, str(time.time()).replace('.', '')))[:225] # max file len on windows/linux is 255 bytes
        
        # create necessary files
        dataset_folder_path = normp(ImputeBenchLabeler.CONF['BENCHMARK_PATH'] + f'/data/{dataset_name}/')
        dataset_file_path = normp(dataset_folder_path + f'/{dataset_name}_normal.txt')
        os.makedirs(dataset_folder_path) # create folder
        timeseries.to_csv(dataset_file_path, sep=' ', index=False, header=None) # create file
        
        try:
            # call the benchmark with those time series and those arguments
            try:
                is_on_windows = os.name == 'nt'
                command = [('wsl ' if is_on_windows else '') + 'mono', 'TestingFramework.exe', 
                           '-alg', algorithms, '-d', dataset_name, '-scen', scenario, 
                           '-nort']
                if not plots:
                    command.append('-novis')
                p = subprocess.check_output(command, cwd=ImputeBenchLabeler.CONF['BENCHMARK_PATH'], 
                                            shell=is_on_windows, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

            # get results
            all_results_folder_path = normp(ImputeBenchLabeler.CONF['BENCHMARK_PATH'] + f'/Results/{scenario}/{dataset_name}/')
            
            all_error_results = []
            for _, dirnames, _ in os.walk(all_results_folder_path): # check all subdirectories of the results folder
                for error in dirnames: 
                    if error in errors: # if the subdirectory contains one of the errors to retrieve
                        results_folder_path = normp(all_results_folder_path + f'/error/{error}/')
                        all_results = []
                        for filename in os.listdir(results_folder_path): # for each file containing results
                            error_result_file_path = normp(results_folder_path + f'/{filename}')
                            results = pd.read_csv(error_result_file_path, header=None, skiprows=1, 
                                                  sep=' ', names=['miss_perc', error]) 
                            with open(error_result_file_path, 'r') as f:
                                results['algorithm'] = f.readline()[2:-1]
                            all_results.append(results)
                    
                        # concat all results for this type of error
                        all_results_df = pd.concat(all_results, axis=0)
                        all_results_df = all_results_df.set_index(['algorithm', 'miss_perc'])
                        all_error_results.append(all_results_df)
            all_error_results_df = pd.concat(all_error_results, axis=1)
        
        finally:
            pass
            # delete generated files
            shutil.rmtree(dataset_folder_path, ignore_errors=True) # remove data set dir
            if delete_results and 'all_results_folder_path' in locals():
                shutil.rmtree(all_results_folder_path, ignore_errors=True) # remove results dir 
        
        # return results
        return all_error_results_df, dataset_name


    # static methods

    def load_labels(dataset_name, properties):
        # TODO
        labels_filename = ImputeBenchLabeler._get_labels_filename(dataset_name)
        clusters_labels = pd.read_csv(labels_filename)
        labels, all_benchmark_results = tsc_utils.get_clusters_labels(clusters_features)
            
        # create correct labels from benchmark_results if the loaded labels are not in the right format        
        labels, algorithms_list = _label_from_benchmark_results(labels, all_benchmark_results, multilabels, 
                                                                multilabels_top_n=multilabels_top_n)

        clusters_features['Label'] = labels
        clusters_features['BenchmarkResults'] = all_benchmark_results

    def save_labels(dataset_name, labels):
        # TODO
        labels_filename = ImputeBenchLabeler._get_labels_filename(dataset_name)
        labels.to_csv(labels_filename, index=False)

    def _get_labels_filename(dataset_name):
        # TODO
        return normp(ImputeBenchLabeler.LABELS_DIR + f'/{dataset_name}{ImputeBenchLabeler.LABELS_APPENDIX}')

    
    # ------------------------------------------------- old

    
    def _reduce_labels_set(self, clusters_features, labels, all_benchmark_results, multilabels, multilabels_top_n, underused_algos_perc):
        # identify algorithms with < X% attribution
        score_matrix = create_algos_score_matrix(clusters_labels=clusters_features)
        ranking_matrix = create_algos_ranking_matrix(score_matrix=score_matrix)
        nb_clusters = ranking_matrix.iloc[0].sum()
        used_perc = ranking_matrix.iloc[:, 0:multilabels_top_n if multilabels else 1].sum(axis=1) / nb_clusters
        used_perc = used_perc.loc[used_perc < underused_algos_perc]
        underused_algos = list(used_perc.index)
        if 'cdrec' in underused_algos:
            underused_algos.extend(['cdrec_k2', 'cdrec_k3'])
        
        # for each cluster that has one of the underused algos as label:
        # replace it by the 2nd best in the BenchmarkResults list
        labels, algorithms_list = _label_from_benchmark_results(labels, all_benchmark_results, multilabels,
                                                                multilabels_top_n=multilabels_top_n,
                                                                algos_to_exclude=underused_algos)
        clusters_features['Label'] = labels
        return clusters_features

    def _label_from_benchmark_results(labels, all_benchmark_results, multilabels, multilabels_top_n=3, algos_to_exclude=[]):
        new_labels = []
        for _, benchmark_results_dict in zip(labels, all_benchmark_results):
            benchmark_results = pd.DataFrame.from_dict(ast.literal_eval(benchmark_results_dict))
            benchmark_results = benchmark_results.rename_axis(index=['algorithm', 'miss_perc'])
            label, algorithms_list = (_get_best_algo(benchmark_results, algos_to_exclude=algos_to_exclude), None) \
                                    if not multilabels else \
                                    _create_multilabels_vector(benchmark_results, top_n=multilabels_top_n, algos_to_exclude=algos_to_exclude)
            new_labels.append(label)
        return new_labels, algorithms_list

    def _get_best_algo(benchmark_results, score_to_measure='rmse', algos_to_exclude=[]):
        # analyze results of benchmark
        best_algo, _ = benchmark.get_highest_benchmark_score(benchmark_results, 
                                                            score_to_measure=score_to_measure,
                                                            algos_to_exclude=algos_to_exclude)

        if 'cdrec_' in best_algo:
            best_algo = 'cdrec'
        return best_algo

    def _create_multilabels_vector(benchmark_results, top_n=3, score_to_measure='rmse', algos_to_exclude=[]):
        algorithms_labels = _create_reduced_multilabels_set(algos_to_exclude) \
                            if len(algos_to_exclude) > 0 else \
                            tsc_utils.ALGORITHMS_LABEL
        vec = np.zeros(len(set(algorithms_labels.values())))
        
        benchmark_results = benchmark_results.groupby('algorithm').mean()
        # drop the intersection btw algos_to_exclude & algos listed in benchmark_results
        algos_to_exclude = list(set(algos_to_exclude) & set(benchmark_results.index))
        benchmark_results_reduced = benchmark_results.drop(benchmark_results.loc[algos_to_exclude].index)
        # keep top-n rows
        top_n_benchmark_results = benchmark_results_reduced.nsmallest(top_n, score_to_measure, keep='all')
        
        for label in top_n_benchmark_results.index:
            label_idx = algorithms_labels[label]
            vec[label_idx] = 1
            
        #                mae      rmse       mse
        #algorithm                              
        #cdrec_k2   0.645027  0.863127  0.749762
        #cdrec_k3   0.693679  0.883453  0.791455
        #dynammo    0.610612  0.791175  0.632809
        #grouse     0.694662  0.888747  0.796800
        #rosl       0.742681  0.943341  0.893658
        #softimp    0.642368  0.822577  0.688262
        #stmvl      0.640512  0.830063  0.692810
        #svdimp     0.672022  0.859401  0.749986
        #svt        0.638844  0.816731  0.668205
        #tenmf      1.653684  2.070520  5.683466
        #trmf       0.649153  0.831884  0.703598
        
        return vec, algorithms_labels

    def _create_reduced_multilabels_set(algos_to_exclude):
        new_algorithms_label = tsc_utils.ALGORITHMS_LABEL.copy()
        algorithms_label_keys = list(new_algorithms_label.keys())

        handled_cdrec = False
        for label_to_excl in algos_to_exclude:
            idx = algorithms_label_keys.index(label_to_excl)
            val = new_algorithms_label[label_to_excl]
            if ('cdrec' in label_to_excl and not handled_cdrec) or 'cdrec' not in label_to_excl:
                if 'cdrec' in label_to_excl:
                    handled_cdrec = True
                for i in range(idx+1, len(algorithms_label_keys)):
                    new_algorithms_label[algorithms_label_keys[i]] -= 1
        for label_to_excl in algos_to_exclude:
            del new_algorithms_label[label_to_excl]
        return new_algorithms_label

    def create_algos_score_matrix(clusters_labels=None, score_to_measure='rmse'):
        # create scores data frame
        score_matrix = pd.DataFrame()
        if clusters_labels is None:
            clusters_labels = pd.read_csv(tsc_utils.CLUSTERS_LABELS_FILENAME)
        for i, benchmark_results_dict in enumerate(clusters_labels['BenchmarkResults']):
            benchmark_results = pd.DataFrame.from_dict(ast.literal_eval(benchmark_results_dict))
            benchmark_results = benchmark_results.rename_axis(index=['algorithm', 'miss_perc'])
            benchmark_results = benchmark_results.groupby('algorithm').mean()[score_to_measure]

            #if i < 10: # tmp
            #    benchmark_results = benchmark_results.drop(['cdrec_k2', 'cdrec_k3', 'dynammo', 'grouse', 'rosl', 'softimp']) # tmp

            missing_algos = [key for key in tsc_utils.ALGORITHMS_LABEL.keys() if key not in benchmark_results.index]
            benchmark_results = benchmark_results.reindex(benchmark_results.index.union(missing_algos))

            benchmark_results.loc['cdrec'] = min(benchmark_results.loc['cdrec_k2'], benchmark_results.loc['cdrec_k3'])
            benchmark_results = benchmark_results.drop(['cdrec_k2', 'cdrec_k3'])

            score_matrix[i] = benchmark_results
            
        return score_matrix

    def create_algos_ranking_matrix(score_to_measure='rmse', score_matrix=None):
        # create scores data frame
        if score_matrix is None:
            score_matrix = create_algos_score_matrix(score_to_measure)
            
        # create ranking matrix
        ranking_matrix = pd.DataFrame(index=tsc_utils.ALGORITHMS_LABEL.keys(), 
                                    columns=range(1, len(tsc_utils.ALGORITHMS_LABEL.keys()) + 1 - 2)).fillna(0)
        ranking_matrix = ranking_matrix.drop(['cdrec_k2', 'cdrec_k3'])
        for col in score_matrix:
            ranking = score_matrix[col].sort_values()
            for i in range(len(ranking)):
                algo = ranking.index[i]
                rank = i + 1
                ranking_matrix.at[algo, rank] += 1

        return ranking_matrix

