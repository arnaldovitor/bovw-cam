import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class BoVW:
    def __init__(self, dictionary: pd.DataFrame, num_cluster: int, kmeans_batch_size: int) -> None:
        """
        Initializes an instance of the Bag-of-Visual-Words (BoVW) class.

        Args:
            dictionary (pd.DataFrame): A DataFrame representing the visual word dictionary.
            num_cluster (int): The number of clusters to be used in KMeans clustering.
            kmeans_batch_size (int): The number of samples to be used in each KMeans iteration.
        """
        self.metadata_columns = ['image_path', 'keypoint_coord_x', 'keypoint_coord_y', 'target']
        self.mini_batch_kmeans = MiniBatchKMeans(
            n_clusters=num_cluster, batch_size=kmeans_batch_size
        ).fit(dictionary.drop(self.metadata_columns, axis=1))

    def _get_single_histogram(self, features: pd.DataFrame) -> tuple:
        """
        Calculates the histogram and cluster indices for a single set of features.

        Args:
            features (pd.DataFrame): A DataFrame representing the features for a single image.

        Returns:
            tuple: A tuple containing the histogram and the cluster indices.
        """
        cluster_idx = self.mini_batch_kmeans.predict(features)
        histogram = np.bincount(cluster_idx, minlength=len(self.mini_batch_kmeans.cluster_centers_))
        return histogram.tolist(), cluster_idx

    def get_feature_vectors(self, extracted_features: pd.DataFrame) -> tuple:
        histogram_columns = [
            f'histogram_{i+1}' for i in range(len(self.mini_batch_kmeans.cluster_centers_))
        ]
        histogram_list = []
        metadata_list = []

        for _, group in extracted_features.groupby('image_path'):
            features = group.drop(columns=self.metadata_columns)
            histogram, cluster_idx = self._get_single_histogram(features)
            metadata = group[self.metadata_columns[:-1]].iloc[0].tolist()
            histogram_list.append(
                [group['image_path'].iloc[0], group['target'].iloc[0]] + histogram
            )
            metadata_list.extend([metadata + [idx] for idx in cluster_idx])

        histogram_columns = ['image_path', 'target'] + histogram_columns
        histogram_dataframe = pd.DataFrame(histogram_list, columns=histogram_columns)
        metadata_columns = self.metadata_columns[:-1] + ['cluster_idx']
        metadata_dataframe = pd.DataFrame(metadata_list, columns=metadata_columns)

        return histogram_dataframe, metadata_dataframe
