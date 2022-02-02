"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
TopologicalFeaturesExtractor.py
@author: @chacungu
"""

from gtda.diagrams import PersistenceEntropy, Scaler
from gtda.homology import VietorisRipsPersistence
from gtda.metaestimators import CollectionTransformer
from gtda.pipeline import Pipeline
from gtda.time_series import TakensEmbedding
import math
from os.path import normpath as normp
import pandas as pd
from sklearn.decomposition import PCA

from FeaturesExtraction.AbstractFeaturesExtractor import AbstractFeaturesExtractor
from Utils.Utils import Utils

class TopologicalFeaturesExtractor(AbstractFeaturesExtractor):
    """
    Singleton class which computes topological features.
    Source: https://giotto-ai.github.io/gtda-docs/0.3.0/notebooks/time_series_classification.html
    """

    FEATURES_FILENAMES_ID = '_topological'
    CONF = Utils.read_conf_file('topologicalfeaturesextractor')


    # constructor

    def __new__(cls, *args, **kwargs):
        if 'caller' in kwargs and kwargs['caller'] == 'get_instance':
            return super(TopologicalFeaturesExtractor, cls).__new__(cls)
        raise Exception('Singleton class cannot be instantiated. Please use the static method "get_instance".')

    def __init__(self, *args, **kwargs):
        super().__init__()


    # public methods

    def extract(self, dataset):
        """
        Extracts and saves as CSV the features of the given data set's time series.
        
        Keyword arguments:
        dataset -- Dataset objects containing the time series from which features must be extracted
        
        Return:
        Updated Dataset object
        """
        timeseries = dataset.load_timeseries(transpose=True)

        print('Extracting topological features on dataset %s.' % dataset.name)
        try:
            features_df = self.extract_from_timeseries(timeseries)
        
            # save features as CSV
            dataset.save_features(self, features_df)
        except Exception as e:
            print(e)
        return dataset

    def extract_from_timeseries(self, timeseries):
        """
        Extracts the given time series' features.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series (each row is a time series)
        
        Return:
        Pandas DataFrame containing the time series' features (each row is a time series' feature vector)
        """
        topological_transfomer = self._get_topological_transfomer(
            timeseries.shape[0], timeseries.shape[1],
            max_time_delay=TopologicalFeaturesExtractor.CONF['MAX_TIME_DELAY'], 
            max_embedding_dim=TopologicalFeaturesExtractor.CONF['MAX_EMBEDDING_DIM'], 
            stride=TopologicalFeaturesExtractor.CONF['STRIDE'], 
            max_pca_components=TopologicalFeaturesExtractor.CONF['MAX_PCA_COMPONENTS'], 
            homology_dimensions=TopologicalFeaturesExtractor.CONF['HOMOLOGY_DIM'],
        )
        try:
            features = topological_transfomer.fit_transform(timeseries.to_numpy().tolist())
            features_df = pd.DataFrame(features, columns=['topological_%i' % i for i in range(1, len(TopologicalFeaturesExtractor.CONF['HOMOLOGY_DIM'])+1)])
            features_df['Time Series ID'] = timeseries.index

            return features_df
        except ValueError as e:
            raise Exception('Topological features cannot be extracted from this data sets\' time series.')

    def save_features(self, dataset_name, features):
        """
        Saves the given features to CSV.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the features belong
        features -- Pandas DataFrame containing the features to save. Each row is a feature's vector.
                    Columns: Time Series ID, Feature 1's name, Feature 2's name, ...
        
        Return: -
        """
        features_filename = self._get_features_filename(dataset_name)
        features.to_csv(features_filename, index=False)

    def load_features(self, dataset):
        """
        Loads the features of the given data set's name.
        
        Keyword arguments: 
        dataset -- Dataset object to which the features belong
        
        Return: 
        Pandas DataFrame containing the data set's features. Each row is a time series feature vector.
        Columns: Time Series ID, Cluster ID, Feature 1's name, Feature 2's name, ...
        """
        # load clusters features
        features_filename = self._get_features_filename(dataset.name)
        features_df = pd.read_csv(features_filename)
        return features_df
    
    # private methods

    def _get_topological_transfomer(self, nb_timeseries, timeseries_length, 
                                    max_time_delay, max_embedding_dim, stride, max_pca_components, homology_dimensions, n_jobs=-1):
        """
        Creates and returns a topological features transformer.
        Adapted from example: https://giotto-ai.github.io/gtda-docs/0.3.0/notebooks/time_series_classification.html
        
        Keyword arguments: 
        nb_timeseries -- number of time series the transformer will handle
        timeseries_length -- length of the time series the transformer will handle
        max_time_delay -- maximum time delay between two consecutive values for constructing one embedded point.
        max_embedding_dim -- maximum dimension of the embedding space.
        stride -- stride duration between two consecutive embedded points.
        max_pca_components -- maximum number of components for PCA dimensionality reduction
        homology_dimensions -- list of dimensions (non-negative integers) of the topological features to be detected.
        n_jobs -- number of workers to use (default: -1 which consists of using all cores available)

        
        Return: 
        Pandas DataFrame containing the data set's features. Each row is a time series feature vector.
        Columns: Time Series ID, Cluster ID, Feature 1's name, Feature 2's name, ...
        """
        #  embedding vectors
        embedding_time_delay = max(min(
            max_time_delay,
            int(timeseries_length / 3)
        ), 1)
        embedding_dimension = max(min(
            max_embedding_dim,
            math.floor( timeseries_length / embedding_time_delay )
        ), 2)
        
        embedder = TakensEmbedding(dimension=embedding_dimension,
                                   time_delay=embedding_time_delay,
                                   stride=stride)

        embedding_shape = (
            nb_timeseries,
            max(1, 1 + int((timeseries_length - (embedding_dimension + (embedding_dimension-1)*(embedding_time_delay-1))) / stride)),
            embedding_dimension
        )
        
        # dimensionality reduction (PCA)
        pca_components = min(
            max_pca_components,
            min(embedding_shape[1], embedding_shape[2])
        )

        print('embedding_time_delay:', embedding_time_delay, 
              ', embedding_dimension:', embedding_dimension, 
              ', stride:', stride, 
              ', embedding_shape:', embedding_shape, 
              ', pca_components:', pca_components) # TODO tmp print
        
        batch_pca = CollectionTransformer(PCA(n_components=pca_components), n_jobs=n_jobs)
        
        #  persistence diagrams
        persistence = VietorisRipsPersistence(homology_dimensions=homology_dimensions, n_jobs=n_jobs)
        
        # scaling
        #scaling = Scaler()
        
        # entropy of the persistence diagrams
        entropy = PersistenceEntropy(normalize=True, nan_fill_value=-10, n_jobs=n_jobs)


        steps = [
            ("embedder", embedder), # list of 2d signals -> 3d embedding vector
            ("pca", batch_pca), # reduces 3rd dim of embedding vector
            ("persistence", persistence), # reduces 2nd dim of PCA vector
            #("scaling", scaling), # produces NaNs which causes the next step to crash.
            ("entropy", entropy) # -> list of 1d vector of len= nb persistence diagrams = len(homology_dimensions)
        ]
        topological_transfomer = Pipeline(steps)
        return topological_transfomer

    def _get_features_filename(self, dataset_name):
        """
        Returns the filename of the features for the given data set's name.
        
        Keyword arguments: 
        dataset_name -- name of the data set to which the features belong
        
        Return: 
        Filename of the features for the given data set's name.
        """
        return normp(
            AbstractFeaturesExtractor.FEATURES_DIR + \
            f'/{dataset_name}{TopologicalFeaturesExtractor.FEATURES_FILENAMES_ID}{AbstractFeaturesExtractor.FEATURES_APPENDIX}')


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