"""
ConFree_kClustering - A system capable to use any derivative of k-Means clustering algorithm without specifying the number of clusters to produce.
eXascale Infolab, University of Fribourg, Switzerland
***
ConFree_kClustering.py
@author: @chacungu
"""

import math
import numpy as np
import pandas as pd

def sklearn_kmeans_helper(k, X):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k).fit_predict(X)


def cluster(clustering_algo, objective_function, data, 
            obj_thresh, init_obj_thresh, sim_cluster_thresh, centroid_dist_thresh,
            k_perc=0.2, security_limit=5, max_iter=5000, id='data'):
    """
    Clusters the given data iteratively.
    
    Keyword arguments:
    clustering_algo -- clustering method that takes as argument K (the objective number of clusters to form) and the samples to clusters. 
                       It returns the index of the cluster each sample belongs to (such as ScikitLearn's KMeans.fit_predict() method).
    objective_function -- objective function to MAXIMIZE that retuns a single score (float)
    data -- list of lists or numpy ndarray or pandas dataframe that contains the data to cluster.
    obj_thresh -- if cluster has a score higher or equal than this, refinement of this data subset stops, cluster is valid and therefore is accepted 
    init_obj_thresh -- if original data has a score higher or equal than this, clustering is not necessary
    sim_cluster_thresh -- score btw centroids of two clusters must be above this to consider them "similar" and consider merging
    centroid_dist_thresh -- score btw a sample and its current cluster's centroid must greater than this to consider moving the 
                            sample to another cluster
    k_perc -- used to compute K by multiplying this percentage by the number of samples. Must be > 0 (default: 0.2).
    security_limit -- number of iterations without new valid cluster before "helping" the clustering algorithm (default: 5)
    max_iter -- maximum number of iterations before a force-stop (default: 10000)
    id -- used to identify the data that is clustered in print messages (default 'data')
    
    Return: 
    List of labels: index of the cluster each sample belongs to
    """
    if isinstance(data, np.ndarray) or isinstance(data, list) and all((isinstance(subdata, list) or isinstance(subdata, np.array)) for subdata in data):
        data = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        pass
    else: raise TypeError('Invalid type for data.')
    
    nb_entries = data.shape[0]
    
    # cluster iteratively the data using the clustering algorithm
    clusters = _iterative_clustering(clustering_algo, objective_function, data, 
                                     obj_thresh, init_obj_thresh, k_perc, security_limit, max_iter, id)

    # merge clusters if score remains high
    merged_clusters = _merging(clusters, objective_function, nb_entries,
                               sim_cluster_thresh, centroid_dist_thresh)

    labels = map(lambda i: i[1], sorted([
        (sid, cid) # sample id, assigned cluster's id
        for cid, cluster in enumerate(merged_clusters)
        for sid in cluster.index
    ], key=lambda i: i[0]))

    return list(labels)


def _iterative_clustering(clustering_algo, objective_function, data, obj_threshold, init_obj_threshold, k_perc, security_limit, max_iter, id='data'):
    """
    Clusters the given data iteratively.
    
    Keyword arguments:
    clustering_algo -- clustering method that takes as argument K (the objective number of clusters to form) and the samples to clusters. 
                       It returns the index of the cluster each sample belongs to (such as ScikitLearn's KMeans.fit_predict() method).
    objective_function -- objective function to MAXIMIZE that retuns a single score (float)
    data -- pandas DataFrame of samples to cluster (each row is a sample)
    obj_threshold -- if cluster has a score higher or equal than this, refinement of this data subset stops, cluster is valid and therefore is accepted 
    init_obj_threshold -- if original data has a score higher or equal than this, clustering is not necessary
    k_perc -- used to compute K by multiplying this percentage by the number of samples. Must be > 0.
    security_limit -- number of iterations without new valid cluster before "helping" the clustering algorithm
    max_iter -- maximum number of iterations before a force-stop
    id -- used to identify the data that is clustered in print messages (default 'data')
    
    Return: 
    List of resulting clusters (each cluster is a Pandas DataFrame of data samples)
    """
    assert k_perc > 0
    stack = [data] # contains the clusters that have not been handled yet
    result = [] # contains the clusters that have been accepted
    security_count = 0
    iter_count = 0
    while len(stack) > 0:

        subdata = stack.pop()
        assert subdata.shape[0] > 0

        threshold = obj_threshold if iter_count > 0 else init_obj_threshold
        if subdata.shape[0] == 1 or objective_function(subdata) >= threshold:
            security_count = 0
            # score is high enough: accept the cluster
            result.append(subdata)

        elif subdata.shape[0] == 2:
            security_count = 0
            # accept the two remaining samples as individual clusters
            result.append(subdata.iloc[0].to_frame().T)
            result.append(subdata.iloc[1].to_frame().T)

        else:
            # score in this cluster is not high enough: cluster
            security_count += 1

            K = max(2, math.floor(subdata.shape[0] * k_perc))
            if security_count >= 15 * security_limit: 
                # even with help the clustering algo couldn't create valid clusters for too many iterations
                # manually split the current cluster in 2 hoping it unstucks the algorithms
                stack.insert(0, subdata.iloc[0::2])
                stack.insert(0, subdata.iloc[1::2])

            else:
                if security_count >= security_limit: 
                    # the clustering algo is having difficulties creating clusters that match the criteria
                    # to unstuck it, increase K to help create valid clusters
                    K += (security_count // (security_limit-1)) * K
                    K = min(subdata.shape[0], K)
                
                labels = clustering_algo(K, subdata)
                
                for cluster_samples_ids in pd.DataFrame(labels).groupby([0]).indices.values():
                    cluster_samples = subdata.iloc[cluster_samples_ids]
                    stack.insert(0, cluster_samples)
        iter_count += 1
        if iter_count >= max_iter:
            print('Max iteration reached for %s! %i clusters did not meet all requirements yet.' % (id, len(stack)))
            result.extend(stack)
            break
    print('clustering subroutine done for %s w/ %i iterations' % (id, iter_count))
    return result

def _merging(clusters, objective_function, nb_entries, sim_cluster_thresh, centroid_dist_thresh):
    """
    Merges clusters if the objective function score remains high enough.
    
    Keyword arguments:
    clusters -- list of resulting clusters (each cluster is a Pandas DataFrame of data samples)
    objective_function -- objective function to MAXIMIZE that is given a cluster (a Pandas DataFrame of data samples) and retuns a 
                          single score (float)
    nb_entries -- number of entries in the dataset the cluster belongs to
    sim_cluster_thresh -- score btw centroids of two clusters must be above this to consider them "similar" and consider merging
    centroid_dist_thresh -- score btw a sample and its current cluster's centroid must greater than this to consider moving the 
                            sample to another cluster
    
    Return:
    Updated clusters: list of updated clusters (each cluster is a Pandas DataFrame of data samples)
    """
    clusters = dict(zip(range(0, len(clusters)), clusters))
    
    # init: compute a centroid for each cluster as well as the score inside each cluster
    all_centroids, all_scores = {}, {}
    for cid, cluster_samples in clusters.items():
        all_centroids[cid] = cluster_samples.mean().to_frame().T
        all_scores[cid] = objective_function(cluster_samples)

    # for each cluster
    for cid in list(clusters.keys()):
        # retrieve the cluster's samples and centroid
        cluster_samples = clusters[cid]
        centroid = all_centroids[cid]

        # compute the score btw the cluster's centroid and all other clusters' centroid and identify a list of similar clusters
        is_cluster_similar = lambda other_cid: \
                                other_cid != cid and \
                                objective_function(pd.concat([centroid, all_centroids[other_cid]])) >= sim_cluster_thresh
        similar_clusters_ids = [other_cid for other_cid in clusters.keys() if is_cluster_similar(other_cid)]

        res = _merging_subroutine(objective_function, clusters, cluster_samples, cid, similar_clusters_ids, 
                                  nb_entries, all_scores, all_centroids)
        merged, clusters, all_centroids, all_scores = res
        if merged:
            del clusters[cid]
            del all_centroids[cid]
            del all_scores[cid]

        # ---
        # we did not found a valid candidate (cluster) to merge with
        # try to move the samples that are the farthest from the cluster's centroid to other clusters
        if not merged and cluster_samples.shape[0] > 1:
            # identify samples that are "far away" from their cluster's centroid
            is_far_from_centroid = lambda item: \
                                        objective_function(pd.concat([centroid, 
                                                                      item[1].to_frame().T])) < centroid_dist_thresh
            farthest_samples = filter(is_far_from_centroid, cluster_samples.iterrows())

            for sid, sample in farthest_samples:

                res = _merging_subroutine(objective_function, clusters, sample.to_frame().T, cid, filter(lambda id: id != cid, clusters.keys()), 
                                          nb_entries, all_scores, all_centroids)
                merged, clusters, all_centroids, all_scores = res
                if merged: 
                    # 1 ts has been moved -> update
                    clusters[cid] = clusters[cid].drop(sid)
                    if clusters[cid].shape[0] > 0:
                        all_centroids[cid] = clusters[cid].mean().to_frame().T
                        all_scores[cid] = objective_function(clusters[cid])
                    else:
                        # no samples are left in the cluster: delete it
                        del clusters[cid]
                        del all_centroids[cid]
                        del all_scores[cid]
                
            
    return list(clusters.values())

def correlation_gain(objective_function, clusters, sample_to_merge, cid, other_cid, nb_entries, all_scores):
    """
    Computes the correlation gain of merging/moving a cluster/sequence with/to a different cluster.
    
    Keyword arguments:
    objective_function -- objective function to MAXIMIZE. Takes a cluster (a Pandas DataFrame of data samples) and retuns a 
                          single score (float)
    clusters -- list of clusters (each cluster is a Pandas DataFrame of samples)
    sample_to_merge -- Pandas DataFrame containing the single sample to try merging
    cid -- id of the clusters from which the sample_to_merge is originating
    other_cid -- id of the other cluster that is a candidate for the merge
    nb_entries -- number of entries in the dataset the cluster belongs to
    all_scores -- dict with keys being clusters 'id and values their current objective function's score
    
    Return:
    Correlation gain.
    """
    phi_union = objective_function(pd.concat([sample_to_merge, clusters[other_cid]]))
    phi_i = all_scores[cid]
    phi_j = all_scores[other_cid]
    corr_gain = (1 / (2*nb_entries)) * (phi_union - ((phi_i * phi_j) / nb_entries))
    return corr_gain

def _merging_subroutine(objective_function, clusters, samples_to_merge, cid, other_clusters_ids, nb_entries, all_scores, all_centroids):
    """
    Searches for a cluster to merge the given samples with.
    
    Keyword arguments:
    objective_function -- objective function to MAXIMIZE that is given a cluster (a Pandas DataFrame of data samples) and retuns a 
                          single score (float)
    clusters -- list of clusters (each cluster is a Pandas DataFrame of samples)
    samples_to_merge -- Pandas DataFrame of samples to try merging (each row is a sample)
    cid -- id of the clusters from which the samples_to_merge are originating
    other_clusters_ids -- list of other clusters' ID that are candidates for a merge
    nb_entries -- number of entries in the dataset the cluster belongs to
    all_scores -- dict with keys being clusters 'id and values their current objective function's score
    all_centroids -- dict with keys being clusters 'id and values their current centroid
    
    Return:
    1. True if a merged occured, False otherwise
    2. Updated clusters: list of updated clusters (each cluster is a Pandas DataFrame of samples)
    3. Updated all_scores
    4. Updated all_centroids
    """
    best_corr_gain, best_cid = 0, None
    # for each cluster
    for other_cid in other_clusters_ids:
        corr_gain = correlation_gain(objective_function, clusters, samples_to_merge, cid, other_cid, nb_entries, all_scores)
        # is this other cluster the best candidate yet for merging?
        if corr_gain > best_corr_gain:
            best_corr_gain = corr_gain
            best_cid = other_cid

    # merge with the best candidate (if we found one)
    merged = False
    if best_cid != None:
        merged = True
        # merge samples with the best_cid's cluster
        merged_cluster_samples = pd.concat([samples_to_merge, clusters[best_cid]])
        clusters[best_cid] = merged_cluster_samples
        all_centroids[best_cid] = merged_cluster_samples.mean().to_frame().T
        all_scores[best_cid] = objective_function(merged_cluster_samples)

    return merged, clusters, all_centroids, all_scores



if __name__ == "__main__":
    import itertools
    
    obj_func = lambda x: np.mean([-((i[0]-j[0])**2 + (i[1]-j[1])**2) for i,j in itertools.combinations(x.to_numpy(),2)])
    l = [[1,2],[1,3],[7,2],[5,3],[6,7],[9,1],[2,6],[5,8],[4,7]]
    print(cluster(
            clustering_algo=sklearn_kmeans_helper, 
            objective_function=obj_func, 
            data=l, 
            obj_thresh=-2.5, 
            init_obj_thresh=-1.5, 
            sim_cluster_thresh=-3, 
            centroid_dist_thresh=1)) 