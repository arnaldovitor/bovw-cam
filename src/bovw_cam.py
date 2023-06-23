from typing import Any

import numpy as np
import pandas as pd


class BoVWCAM:
    def __init__(self, classifier: Any) -> None:
        self.classifier = classifier

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
        self, dictionary_histogram: pd.DataFrame, target_column_name: str
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
                if column_name not in (target_column_name, 'image_path'):
                    matrix_row.append(
                        dictionary_histogram[column_name].corr(
                            dictionary_histogram['temp_target'], method='spearman'
                        )
                    )
            correlation_matrix.append(matrix_row)

        dictionary_histogram.drop('temp_target', axis=1, inplace=True)
        return correlation_matrix
