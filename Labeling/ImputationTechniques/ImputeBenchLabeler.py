"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
XXXXX
***
ImputeBenchLabeler.py
@author: @XXXXX
"""

import ast
import numpy as np
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
    Singleton labeler class which uses the ImputeBench benchmark. Provides methods to label time series and handle those labels.

    ImputeBench is presented in "Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series"
    by Khayati et al. in 2020.
    """

    LABELS_DIR = normp('./Labeling/ImputationTechniques/labels/')
    LABELS_FILENAMES_ID = '_ibl'
    CONF = Utils.read_conf_file('imputebenchlabeler')

    # create necessary directories if not there yet
    Utils.create_dirs_if_not_exist([LABELS_DIR])


    # constructor

    def __new__(cls, *args, **kwargs):
        if 'caller' in kwargs and kwargs['caller'] == 'get_instance':
            return super(ImputeBenchLabeler, cls).__new__(cls)
        raise Exception('Singleton class cannot be instantiated. Please use the static method "get_instance".')

    def __init__(self, *args, **kwargs):
        super().__init__()


    # public methods

    def label_all_datasets(self, datasets):
        """
        Labels each cluster from the given list of data sets using the ImputeBench benchmark.
        
        Keyword arguments:
        datasets -- List of Dataset objects containing time series to label
        
        Return:
        List of updated Dataset objects
        """
        updated_datasets = []

        # for each data set
        for dataset in datasets:
            dataset = self.label(dataset)
            updated_datasets.append(dataset)
        return updated_datasets

    def label(self, dataset):
        """
        Labels each cluster from the given data set using the ImputeBench benchmark.
        
        Keyword arguments:
        dataset -- Dataset object containing time series to label
        
        Return:
        Updated Dataset object
        """
        tmp_labels = []
        
        # load time series
        timeseries = dataset.load_timeseries(transpose=True)

        print('Labeling %i clusters of %s.' % (dataset.load_cassignment(dataset.clusterer)['Cluster ID'].nunique(), dataset.name)) # TODO tmp print

        # for each cluster
        for cluster, cluster_id, _ in dataset.yield_all_clusters(timeseries):
            print('Running benchmark for cluster %i (%s)' % (cluster_id, dataset.name))
            # label the cluster's time series
            benchmark_results = self._label_cluster(timeseries, cluster, cluster_id)
            tmp_labels.append((cluster_id, benchmark_results))
        tmp_labels_df = pd.DataFrame(tmp_labels, columns=['Cluster ID', 'Benchmark Results'])
        
        # save labels
        dataset.save_labels(self, tmp_labels_df)
        return dataset

    def get_default_properties(self):
        """
        Returns a dict containing the default properties that labels generated with this labeler have.
        
        Keyword arguments: -
        
        Return: 
        Dict containing the default properties that labels generated with this labeler have.
        """
        return ImputeBenchLabeler.CONF['LBL_PROPERTIES']

    def save_labels(self, dataset_name, labels):
        """
        Saves the given labels to CSV.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the labels belong
        labels -- Pandas DataFrame containing the labels to save. Two columns: Cluster ID, Benchmark Results.
        
        Return: -
        """
        labels_filename = self._get_labels_filename(dataset_name)
        labels.to_csv(labels_filename, index=False)

    def load_labels(self, dataset, properties):
        """
        Loads the labels of the given data set's name and defined by the specified properties.
        
        Keyword arguments: 
        dataset -- Dataset object to which the labels belong
        properties -- dict specifying the labels' properties
        
        Return: 
        1. Pandas DataFrame containing the data set's labels. Two columns: Time Series ID and Label.
        2. List of all possible labels value
        """
        # load clusters labels
        labels_filename = self._get_labels_filename(dataset.name)
        clusters_labels = pd.read_csv(labels_filename, index_col='Cluster ID')
        
        # create labels from benchmark_results
        clusters_labels_df, algorithms_list = self._get_labels_from_bench_res(clusters_labels, properties)
        clusters_labels_dict = clusters_labels_df.set_index('Cluster ID').to_dict()['Label']

        # propagate clusters' label to their time series
        timeseries_labels = []

        for _, row in dataset.load_cassignment(dataset.clusterer).iterrows():
            tid = row['Time Series ID']
            cid = row['Cluster ID']
            label = clusters_labels_dict[cid]
            timeseries_labels.append((tid, label))

        timeseries_labels_df = pd.DataFrame(timeseries_labels, columns=['Time Series ID', 'Label'])
        return timeseries_labels_df, algorithms_list

    def create_algos_score_matrix(self, all_benchmark_results, score_to_measure):
        """
        Creates scores DataFrame from the ImputeBench benchmark results.
        
        Keyword arguments: 
        all_benchmark_results -- Pandas DataFrame with Cluster ID as index and the corresponding ImputeBench 
                                 benchmark results as the only column
        score_to_measure -- error to minimize when selecting relevant algorithms
        
        Return:
        Scores DataFrame from the ImputeBench benchmark results. Index: algorithms' names. 
        Columns: Cluster ID. Values: measured error when running the ImputeBench benchmark.
        """
        raise Exception('Old code that should not be used anymore.')
        # create scores data frame
        score_matrix = pd.DataFrame()
        for i, benchmark_results_dict in enumerate(all_benchmark_results['Benchmark Results']):
            # convert bench_res to DataFrame
            benchmark_results = self._convert_and_aggregate_bench_res(
                benchmark_results_dict,
                aggregation_strat=ImputeBenchLabeler.CONF['BENCH_RES_AGGREGATION_STRATEGY']
            )[score_to_measure]

            # identify possible missing algos
            missing_algos = [algo for algo in ImputeBenchLabeler.CONF['ALGORITHMS_LIST'] if algo not in benchmark_results.index]
            benchmark_results = benchmark_results.reindex(benchmark_results.index.union(missing_algos))

            # merge all versions of cdrec (keep the best one)
            benchmark_results.loc['cdrec'] = min(benchmark_results.loc['cdrec_k2'], benchmark_results.loc['cdrec_k3'])
            benchmark_results = benchmark_results.drop(['cdrec_k2', 'cdrec_k3'])

            score_matrix[i] = benchmark_results
            
        return score_matrix

    def create_algos_ranking_matrix(self, score_matrix, score_to_measure):
        """
        Creates scores DataFrame from the score_matrix created from the ImputeBench benchmark results.
        
        Keyword arguments: 
        score_matrix -- Scores DataFrame from the ImputeBench benchmark results. Index: algorithms' names. 
                        Columns: Cluster ID. Values: measured error when running the ImputeBench benchmark.
        score_to_measure -- error to minimize when selecting relevant algorithms
        
        Return:
        Ranking DataFrame from the ImputeBench benchmark results. Index: algorithms' names. 
        Columns: Position at which an algorithm is recommended. Values: Number of clusters labeled with corresponding 
        algorithm at Nth ranking.
        """       
        # create ranking matrix
        ranking_matrix = pd.DataFrame(index=ImputeBenchLabeler.CONF['ALGORITHMS_LIST'], 
                                      columns=range(1, len(ImputeBenchLabeler.CONF['ALGORITHMS_LIST']) + 1)).fillna(0)
        for col in score_matrix:
            ranking = score_matrix[col].sort_values()
            for i in range(len(ranking)):
                algo = ranking.index[i]
                rank = i + 1
                ranking_matrix.at[algo, rank] += 1

        return ranking_matrix


    # private methods

    def _label_cluster(self, all_timeseries, cluster, cluster_id):
        """
        Labels the given cluster using the ImputeBench benchmark.
        
        Keyword arguments:
        all_timeseries -- DataFrame containing all the time series of the cluster's data set (each row is a time series)
        cluster -- DataFrame containing only the time series of the cluster (each row is a time series)
        cluster_id -- cluster ID
        
        Return:
        Dict containing the ImputeBench benchmark results
        """   
        # select the sequences to give to the benchmark (either the whole cluster or the whole data set)
        if ImputeBenchLabeler.CONF['TS_SELECTION_FOR_BCHMK'] == 'DATASET':
            sequences_to_use = pd.concat([
                cluster.iloc[0].to_frame(), # 1st seq of cluster we want to label (benchmark will try to reconstruct this one)
                all_timeseries # all time series in the data set
            ])
        elif ImputeBenchLabeler.CONF['TS_SELECTION_FOR_BCHMK'] == 'CLUSTER':
            sequences_to_use = cluster
        elif ImputeBenchLabeler.CONF['TS_SELECTION_FOR_BCHMK'] == 'RDM_FROM_DATASET':
            # select all data set's sequences except the ones from the cluster we want to label
            available_timeseries = pd.concat([all_timeseries, cluster]).drop_duplicates(keep=False)
            # number of time series to fed to the benchmark in addition to the sequence we want to label and a second sequence from its cluster
            N_to_sample = min(ImputeBenchLabeler.CONF['NB_TS_FOR_BCHMK'], available_timeseries.shape[0])

            sequences_to_use = pd.concat([
                # 1st (and 2nd if cluster has at least 2 series) seq of cluster we want to label (benchmark will try to reconstruct this one)
                *[cluster.iloc[i].to_frame().T for i in range(0, int(cluster.shape[0] > 1)+1)],
                # N sequences from the same data set but not the same cluster
                available_timeseries.sample(N_to_sample, replace=False) 
            ])

            if sequences_to_use.shape[0] < 5:
                nb_seq_to_add = 5 - sequences_to_use.shape[0]
                if cluster.shape[0] - 2 >= nb_seq_to_add:
                    sequences_to_use = pd.concat([
                        sequences_to_use,
                        *[cluster.iloc[-i-1].to_frame().T for i in range(nb_seq_to_add)]
                    ])
                else: raise Exception('The data set does not have enough time series for the ImputeBench benchmark to run properly.')


            print(all_timeseries.shape, cluster.shape, sequences_to_use.shape) # TODO tmp print

        else: raise Exception('Invalid strategy for time series selection for benchmark.')

        # call benchmark on the cluster's time series
        alg_algx_cmd = self._get_benchmark_alg_cmd(ImputeBenchLabeler.CONF['ALGORITHMS_LIST'])
        benchmark_results, _ = self._run_benchmark(sequences_to_use.T, 
                                                   alg_algx_cmd=alg_algx_cmd,
                                                   scenario=ImputeBenchLabeler.CONF['BENCHMARK_SCENARIO'], 
                                                   errors=ImputeBenchLabeler.CONF['BENCHMARK_ERRORS'],
                                                   id=cluster_id,
                                                   plots=False,
                                                   delete_results=True)

        return benchmark_results.to_dict()

    def _get_benchmark_alg_cmd(self, algorithms):
        """
        Returns a tuple containing first '-alg' and followed by the algorithms names in the right format to be included as is in
        the command to run the benchmark.
        
        Keyword arguments:
        algorithms -- tuple containing first '-alg' and followed by the argument's value(s) 
                      as strings (e.g. ('-alg', 'all') or ('-alg', 'cdrec,stmvl') )

        Return:
        tuple containing first '-alg' and followed by the algorithms names
        """
        # transform arguments to command line compatible arguments
        if isinstance(algorithms, list):
            algorithms = ','.join(algorithms)
        return '-alg', algorithms

    def _run_benchmark(self, timeseries, alg_algx_cmd, scenario, errors, id, plots=False, delete_results=True):
        """
        Runs the ImputeBench benchmark on given time series and scenario. 
        If on Windows, wsl must be installed.
        
        Keyword arguments:
        timeseries -- DataFrame containing the time series (each column is a time series)
        alg_algx_cmd -- tuple containing first either '-alg' or '-algx' and followed by the argument's value(s)
                        as strings (e.g. ('-algx', 'svdimp', '4'), or ('-alg', 'all') )
        scenario -- string of the scenario's name to run - ONLY "miss_perc" can be used as of now
        errors -- list of string for the errors' names to get in the returned DataFrame
        id -- custom id to incorporate in the temporary files/folders names
        plots -- creates reconstruction plots if True (default True)
        delete_results -- deletes the results folder if True (default True)
        
        Return:
        1. DataFrame containing the different scores. Multi index: "algorithms" & "scenario".
        2. Temporary file/folders name used for the benchmark
        """ 
            
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
                           *alg_algx_cmd, '-d', dataset_name, '-scen', scenario, 
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

    def _get_labels_filename(self, dataset_name):
        """
        Returns the filename of the labels for the given data set's name.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the labels belong
        
        Return: 
        Filename of the labels for the given data set's name.
        """
        return normp(
            ImputeBenchLabeler.LABELS_DIR + \
            f'/{dataset_name}{ImputeBenchLabeler.LABELS_FILENAMES_ID}{AbstractLabeler.LABELS_APPENDIX}')

    def _convert_bench_res_to_df(self, benchmark_results_dict):
        """
        Converts the benchmark results dict to a Pandas DataFrame.
        
        Keyword arguments: 
        benchmark_results_dict -- dict containing the ImputeBench benchmark results
        
        Return:
        Benchmark results dict converted to a Pandas DataFrame. Pandas DataFrame containing the different scores from the ImputeBench benchmark. 
        Multi index: "algorithms" & "scenario".
        """
        benchmark_results = pd.DataFrame.from_dict(ast.literal_eval(benchmark_results_dict))
        benchmark_results = benchmark_results.rename_axis(index=['algorithm', 'miss_perc'])
        return benchmark_results

    def _aggregate_bench_res(self, benchmark_results, index_to_agg, agg_strat):
        """
        Aggregates the benchmark results using the specified aggregation strategy.
        
        Keyword arguments: 
        benchmark_results -- Pandas DataFrame containing the different scores from the ImputeBench benchmark. 
                             Multi index: "algorithms" & "scenario".
        index_to_agg -- benchmark results' indices (e.g. [0,1,2]) to aggregate (level 1: i.e. index of missing percentages)
        agg_strat -- aggregation strategy for the benchmark results
        
        Return:
        Aggregated benchmark results. Index: 'algorithm', Columns: errors.
        """
        assert isinstance(index_to_agg, list) and len(index_to_agg) >= 0 and max(index_to_agg) < len(benchmark_results.index.levels[1].tolist())
        real_index_to_agg = pd.Series(benchmark_results.index.levels[1].tolist())[index_to_agg].tolist()

        if len(index_to_agg) == 1:
            return benchmark_results.loc[(slice(None), real_index_to_agg[0]), :].droplevel('miss_perc')

        df_groupby = benchmark_results.loc[(slice(None), real_index_to_agg), :].groupby('algorithm')
        if agg_strat == 'mean':
            return df_groupby.mean()
        elif agg_strat == 'median':
            return df_groupby.median()
        elif agg_strat == 'sum':
            return df_groupby.sum()
        elif agg_strat == 'min':
            return df_groupby.min()
        elif agg_strat == 'max':
            return df_groupby.max()
        else:
            raise Exception('Invalid ImputeBench results\' aggregatin strategy.')

    def _get_ranking_from_bench_res(self, benchmark_results, ranking_strat, ranking_strat_params, error_to_minimize, algos_to_exclude, return_scores=False):
        """
        Aggregates the benchmark results using the specified aggregation strategy and then ranks the algorithms (classes).
        
        Keyword arguments: 
        benchmark_results -- Pandas DataFrame containing the different scores from the ImputeBench benchmark. 
                             Multi index: "algorithms" & "scenario".
        ranking_strat -- aggregation and ranking strategy to use to get from the ImputeBench benchmark results to a list 
                         of algorithms (ordered from best to worse)
        ranking_strat_params -- dictionary of parameters for the ranking_strat
        error_to_minimize -- error to minimize when selecting relevant algorithms
        algos_to_exclude -- list of algorithms to exclude from the returned list
        return_scores -- True if a second list of scores should be returned in addition to the ranked algorithms, False otherwise (default False)
        
        Return:
        If return_scores is set to False (default): ranked (from best to worse) list of algorithms (classes).
        If return_scores is set to True: pandas DataFrame with ranked (from best to worse) list of algorithms (classes) as index
          and columns that give information about those algorithms (columns depend on the ranking_strat).
        """
        
        ranked_algos = None
        if ranking_strat == 'simple':
            # one aggregation gives the final list
            agg_bench_res = self._aggregate_bench_res(benchmark_results, 
                                                      index_to_agg=ranking_strat_params['index_to_agg'], 
                                                      agg_strat=ranking_strat_params['agg_strat'])
            sorted_bench_scores = agg_bench_res.sort_values(error_to_minimize, ascending=True)
            ranked_algos = sorted_bench_scores.loc[:, [error_to_minimize]]
        elif ranking_strat == 'voting':
            # multiple aggregations -> majority voting to get the final list
            algos_cstm_scores = {}
            algos_list = benchmark_results.index.levels[0].tolist()
            algos_rmse = {algo: ([], []) for algo in algos_list}
            for params in ranking_strat_params.values():
                agg_bench_res = self._aggregate_bench_res(benchmark_results, 
                                                          index_to_agg=params['index_to_agg'], 
                                                          agg_strat=params['agg_strat'])
                sorted_bench_scores_tmp = agg_bench_res.sort_values(error_to_minimize, ascending=True)
                ranked_algos_tmp = sorted_bench_scores_tmp.index.tolist()

                for i, algo in enumerate(ranked_algos_tmp):
                    algos_rmse[algo][0].append( agg_bench_res.loc[algo, error_to_minimize] )
                    algos_rmse[algo][1].append( i+1 )
                    try:
                        algos_cstm_scores[algo] += i
                    except:
                        algos_cstm_scores[algo] = i
            # the algorithm the most often at the beginning of the lists is the best
            ranked_algos = list(dict(sorted(algos_cstm_scores.items(), key=lambda item: item[1])).keys())
            compute_weighted_avg = lambda algo: sum(rmse * weight for rmse, weight in zip(algos_rmse[algo][0], algos_rmse[algo][1])) / \
                                                sum(algos_rmse[algo][1])
            ranked_algos = pd.DataFrame(data={
                'algorithms': ranked_algos, 
                'weighted average %s' % error_to_minimize: [compute_weighted_avg(algo) for algo in ranked_algos],
                'average %s' % error_to_minimize: [sum(algos_rmse[algo][0]) / len(algos_rmse[algo][0]) for algo in ranked_algos],
                'average rank': [sum(algos_rmse[algo][1]) / len(algos_rmse[algo][1]) for algo in ranked_algos],
                'ranks': [algos_rmse[algo][1] for algo in ranked_algos]})
            ranked_algos = ranked_algos.set_index('algorithms')
        else: raise Exception('Invalid ranking strategy.')
        
        # remove algorithms listed in algos_to_exclude
        ranked_algos = ranked_algos[~ranked_algos.index.isin(algos_to_exclude)]
            
        # replace cdrec_k2 and _k3 by cdrec
        if 'cdrec_k2' in ranked_algos.index and 'cdrec_k3' in ranked_algos.index:
            idx1 = ranked_algos.index.get_loc('cdrec_k2')
            idx2 = ranked_algos.index.get_loc('cdrec_k3')
            ranked_algos = ranked_algos.rename({ranked_algos.index[min(idx1, idx2)]: 'cdrec'}) # keep the best of the 2
            ranked_algos = ranked_algos.drop(ranked_algos.index[max(idx1, idx2)], axis=0) # remove the worse of the 2
        elif 'cdrec_k2' in ranked_algos.index or 'cdrec_k3' in ranked_algos.index:
            idx = ranked_algos.index.get_loc('cdrec_k3') if 'cdrec_k3' in ranked_algos.index else ranked_algos.index.get_loc('cdrec_k2')
            ranked_algos = ranked_algos.rename({ranked_algos.index[idx]: 'cdrec'})

        if not return_scores:
            return ranked_algos.index.tolist()
        else:
            return ranked_algos

    def _get_labels_from_bench_res(self, all_benchmark_results, properties):
        """
        Uses the ImputeBench benchmark results to generate clusters' labels.
        
        Keyword arguments: 
        all_benchmark_results -- Pandas DataFrame with Cluster ID as index and the corresponding ImputeBench 
                                 benchmark results as the only column
        properties -- dict specifying the labels' properties
        
        Return:
        1. Pandas DataFrame containing the clusters' labels. Two columns: Cluster ID, Label.
        2. list of all possible labels values
        """
        # identify algorithms to exclude from labels list if some reduction threshold has been specified
        algos_to_exclude = self._get_algos_to_exclude(all_benchmark_results, properties) \
                            if properties['reduction_threshold'] > 0.0 else []

        # get each cluster's label from their benchmark results
        clusters_labels = []
        for cid, bench_res in all_benchmark_results.iterrows():
            # convert bench_res to DataFrame
            benchmark_results = self._convert_bench_res_to_df(bench_res.values[0])
            # get a ranked list of algorithms for this cluster (from best to worse)
            ranking_strat = ImputeBenchLabeler.CONF['BENCH_RES_AGG_AND_RANK_STRATEGY']
            ranked_algos_for_cid = self._get_ranking_from_bench_res(
                benchmark_results,
                ranking_strat=ranking_strat,
                ranking_strat_params=ImputeBenchLabeler.CONF['BENCH_RES_AGG_AND_RANK_STRATEGY_PARAMS'][ranking_strat],
                error_to_minimize=ImputeBenchLabeler.CONF['BENCHMARK_ERROR_TO_MINIMIZE'],
                algos_to_exclude=algos_to_exclude
            )
            
            # create label from benchmark results
            if properties['type'] == 'regression':
                # todo implement regression
                raise Exception('Regression not implemented yet')
            elif properties['type'] == 'multilabels':
                top_n = properties['multi_labels_nb_rel']
                label = self._get_multilabels_vector(ranked_algos_for_cid, top_n)
            else: # monolabels
                label = ranked_algos_for_cid[0] # take the best
                                      
            clusters_labels.append((cid, label))

        algorithms_list = [algo for algo in ImputeBenchLabeler.CONF['ALGORITHMS_LIST'] if algo not in algos_to_exclude]
        clusters_labels_df = pd.DataFrame(clusters_labels, columns=['Cluster ID', 'Label'])
        return clusters_labels_df, algorithms_list

    def _get_algos_to_exclude(self, all_benchmark_results, properties):
        """
        Returns the list of algorithms for which the original clusters' attribution is lower than the 'reduction_threshold'
        threshold specified in the properties. This is the list of algorithms to exclude from the set of labels.
        
        Keyword arguments: 
        all_benchmark_results -- Pandas DataFrame with Cluster ID as index and the corresponding ImputeBench 
                                 benchmark results as the only column
        properties -- dict specifying the labels' properties
        
        Return:
        List of algorithms to exclude from the set of labels
        """
        # create ranking matrix
        score_matrix = self.create_algos_score_matrix(all_benchmark_results, 
                                                      ImputeBenchLabeler.CONF['BENCHMARK_ERROR_TO_MINIMIZE'])
        ranking_matrix = self.create_algos_ranking_matrix(score_matrix, 
                                                          ImputeBenchLabeler.CONF['BENCHMARK_ERROR_TO_MINIMIZE'])
        
        nb_clusters = ranking_matrix.iloc[0].sum()
        
        # identify algorithms with < X% clusters' attribution
        used_perc = ranking_matrix.iloc[:, 0 : properties['multi_labels_nb_rel'] if properties['type'] == 'multilabels' else 1]\
                                  .sum(axis=1) / nb_clusters
        print(used_perc)
        used_perc = used_perc.loc[used_perc < properties['reduction_threshold']]
        algos_to_exclude = list(used_perc.index)
        return algos_to_exclude

    def _get_multilabels_vector(self, ranked_algos_for_cid, top_n):
        """
        Creates and returns a multi-labels vector from the ImputeBench benchmark.
        
        Keyword arguments: 
        ranked_algos_for_cid -- ranked (from best to worse) list of algorithms (classes).
        top_n -- top N algorithms to consider relevant in a multi-labels vector
        
        Return:
        Multi-labels vector where relevant algorithm's bit is set to 1
        """
        algorithms_list = [algo for algo in ImputeBenchLabeler.CONF['ALGORITHMS_LIST'] if algo not in algos_to_exclude]
        vec = np.zeros(len(algorithms_list))
        
        for label in ranked_algos_for_cid[:top_n]:
            label_idx = algorithms_list.index(label)
            vec[label_idx] = 1
        
        return vec

    def _get_best_algo(self, aggregated_benchmark_results, score_to_measure, algos_to_exclude):
        """
        Returns the best algorithm's name from the ImputeBench benchmark.
        
        Keyword arguments: 
        aggregated_benchmark_results -- Aggregated Pandas DataFrame containing the score of each algorithm from the 
                                        ImputeBench benchmark.
        score_to_measure -- error to minimize when selecting relevant algorithms
        algos_to_exclude -- list of algorithms to exclude
        
        Return:
        Best algorithm's name from the ImputeBench benchmark.
        """
        # analyze results of benchmark
        best_algo, _ = self._get_highest_bench_score(aggregated_benchmark_results, score_to_measure, algos_to_exclude)

        if 'cdrec_' in best_algo:
            best_algo = 'cdrec'
        return best_algo

    def _get_highest_bench_score(self, aggregated_benchmark_results, score_to_measure, algos_to_exclude):
        """
        Returns the best algorithm name and its error from the benchmark results.
        
        Keyword arguments:
        aggregated_benchmark_results -- Aggregated Pandas DataFrame containing the score of each algorithm from the 
                                        ImputeBench benchmark.
        score_to_measure -- name of the score to measure
        algos_to_exclude -- list of algorithms names that shouldn't be returned
        
        Return:
        1. name of the algorithm that gave the lowest error on the benchmark
        2. lowest error
        """        
        algos_to_exclude = list(set(algos_to_exclude) & set(aggregated_benchmark_results.index))
        aggregated_benchmark_results = aggregated_benchmark_results.drop(aggregated_benchmark_results.loc[algos_to_exclude].index)
        
        return aggregated_benchmark_results[score_to_measure].idxmin(), aggregated_benchmark_results[score_to_measure].min()


    # static methods

    @classmethod
    def get_instance(cls):
        """
        Returns the single instance of this class.
        
        Keyword arguments: - (no args required, cls is provided automatically since this is a classmethod)
        
        Return: 
        Single instance of this class.
        """
        return super().get_instance(cls)