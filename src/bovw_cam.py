import cv2
import numpy as np
import pandas as pd

from src.classifier_interface import ClassifierInterface


class BoVWCAM:
    def __init__(
        self, classifier: ClassifierInterface, dictionary_histograms: pd.DataFrame
    ) -> None:
        if not isinstance(classifier, ClassifierInterface):
            raise Exception(
                'Exception: Invalid classifier type. Expected an instance of ClassifierInterface.'
            )

        self.classifier = classifier
        self.correlation_matrix = self._calculate_correlation_matrix(dictionary_histograms)

    def _read_image(self, image_path: str) -> np.ndarray:
        """
        Read an image from the specified path.

        Args:
            image_path (str): The path to the image.

        Returns:
            np.ndarray: The image as a numpy array.
        """
        return cv2.imread(image_path)

    def _max_pooling_2d(
        self,
        image: np.ndarray,
        kernel_size: int,
        stride: int,
        padding: int = 0,
    ) -> np.ndarray:
        """Apply 2D max pooling to an image.

        Args:
            image (np.ndarray): Input image.
            kernel_size (int): Size of the pooling kernel.
            stride (int): Stride value for pooling.
            padding (int, optional): Amount of padding. Defaults to 0.
        Raises:
            ValueError: If an invalid pooling type is provided.

        Returns:
            np.ndarray: Resulting pooled image.
        """
        image = np.pad(image, padding, mode='constant')

        output_height = (image.shape[0] - kernel_size) // stride + 1
        output_width = (image.shape[1] - kernel_size) // stride + 1

        pooled_shape = (output_height, output_width, kernel_size, kernel_size)
        pooled_strides = (
            stride * image.strides[0],
            stride * image.strides[1],
            image.strides[0],
            image.strides[1],
        )

        pooled_image = np.lib.stride_tricks.as_strided(image, pooled_shape, pooled_strides)

        return pooled_image.max(axis=(2, 3))

    def _calculate_correlation_matrix(
        self, dictionary_histogram: pd.DataFrame, target_column_name: str = 'target'
    ) -> list:
        """Calculate the correlation matrix for a given dictionary histogram.

        Args:
            dictionary_histogram (pd.DataFrame): Dictionary histogram.
            target_column_name (str): Name of the target column.

        Returns:
            list: Correlation matrix.
        """
        correlation_matrix = []
        unique_target_values = dictionary_histogram[target_column_name].unique()

        for target_value in unique_target_values:
            matrix_row = []
            dictionary_histogram['temp_target'] = (
                dictionary_histogram[target_column_name] == target_value
            ).astype(int)

            for column_name in dictionary_histogram.columns:
                if column_name not in (target_column_name, 'temp_target', 'image_path'):
                    matrix_row.append(
                        dictionary_histogram[column_name].corr(
                            dictionary_histogram['temp_target'], method='spearman'
                        )
                    )
            correlation_matrix.append(matrix_row)

        dictionary_histogram.drop('temp_target', axis=1, inplace=True)
        return correlation_matrix

    def _get_histogram_from_image(
        self, image_path: str, split_histograms: pd.DataFrame
    ) -> np.ndarray:
        """
        Retrieve the histogram from an image.

        Args:
            image_path (str): The path to the image.
            split_histograms (pd.DataFrame): Dataframe containing image histograms.

        Returns:
            np.ndarray: The histogram of the image as a numpy array.
        """
        image_histogram = split_histograms[split_histograms['image_path'] == image_path]
        return image_histogram.drop(['target', 'image_path'], axis=1).to_numpy().reshape(1, -1)

    def _get_keypoints_from_image(
        self, image_path: str, split_metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Retrieve the keypoints from an image.

        Args:
            image_path (str): The path to the image.
            split_metadata (pd.DataFrame): Dataframe containing image metadata.

        Returns:
            pd.DataFrame: Dataframe containing the keypoints of the image.
        """
        return split_metadata[split_metadata['image_path'] == image_path]

    def get_class_activation_map(
        self,
        image_path: str,
        split_histograms: pd.DataFrame,
        split_metadata: pd.DataFrame,
        max_pool_params: dict,
    ) -> np.ndarray:
        """
        Generate the class activation map for an image.

        Args:
            image_path (str): The path to the image.
            split_histograms (pd.DataFrame): Dataframe containing image histograms.
            split_metadata (pd.DataFrame): Dataframe containing image metadata.
            max_pool_params (dict): Parameters for max pooling.

        Returns:
            np.ndarray: The class activation map as a numpy array.
        """
        image = self._read_image(image_path)

        heatmap = np.zeros((image.shape[0], image.shape[1]))
        sample_histogram = self._get_histogram_from_image(image_path, split_histograms)
        image_keypoints = self._get_keypoints_from_image(image_path, split_metadata)

        predicted_class = self.classifier.predict(sample_histogram)

        for _, row in image_keypoints.iterrows():
            heatmap[row['keypoint_coord_y'], row['keypoint_coord_x']] = self.correlation_matrix[
                predicted_class
            ][row['cluster_idx']]

        heatmap = (heatmap + 1e-3) * 255
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        heatmap = self._max_pooling_2d(
            heatmap, max_pool_params['kernel_size'], max_pool_params['stride']
        )
        return cv2.GaussianBlur(heatmap, (9, 9), 0)
