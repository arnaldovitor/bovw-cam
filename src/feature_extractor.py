import itertools
import os
from typing import Any

import cv2
import numpy as np
import pandas as pd


class FeatureExtractor:
    def __init__(self) -> None:
        self.sift = cv2.SIFT_create()
        self.dictionary = pd.DataFrame()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()

    def _compute_sift_features(self, image: np.ndarray) -> tuple:
        """Compute SIFT keypoints and descriptors for an image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            tuple: A tuple containing the keypoints and descriptors.
        """
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def _read_image(self, image_path: str) -> np.ndarray:
        """Read an image from the given path.

        Args:
            image_path (str): The path to the image.

        Returns:
            np.ndarray: The image as a NumPy array.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The image '{image_path}' does not exist.")
        return cv2.imread(image_path)

    def _get_sample_descriptors(
        self, descriptors: Any, dictionary_random_sample_size: float
    ) -> list:
        """Get a random sample of descriptors.

        Args:
            descriptors (Any): The descriptors to sample from.
            dictionary_random_sample_size (float): The fraction of descriptors to sample.

        Returns:
            list: The random sample of descriptors.
        """
        k = int(len(descriptors) * dictionary_random_sample_size)
        return list(itertools.islice(descriptors, k))

    def _sift_features_to_dataframe(
        self, image_path: str, keypoints: Any, descriptors: Any
    ) -> pd.DataFrame:
        """Convert SIFT keypoints and descriptors to a pandas DataFrame.

        Args:
            image_path (str): The path of the image.
            keypoints (Any): The keypoints of the image.
            descriptors (Any): The descriptors of the image.

        Returns:
            pd.DataFrame: A DataFrame containing the SIFT features.
        """
        sift_features = []
        for keypoint, descriptor in zip(keypoints, descriptors, strict=True):
            feature = {
                'image_path': image_path,
                'keypoint_coord_x': keypoint.pt[0],
                'keypoint_coord_y': keypoint.pt[1],
            }

            for i, value in enumerate(descriptor):
                feature[f'feature_{i+1}'] = value

            sift_features.append(feature)

        return pd.DataFrame.from_records(sift_features)

    def _create_simple_split(
        self,
        split_path: str,
        dictionary_split: bool = False,
        dictionary_random_sample_size: float = 0.2,
    ) -> pd.DataFrame:
        """Create a simple train/test split.

        Args:
            split_path (str): The path to the split directory.

        Returns:
            pd.DataFrame: A DataFrame containing the SIFT features of the split.
        """
        split_dataframe = pd.DataFrame()
        for image_entry in os.scandir(split_path):
            if image_entry.is_file():
                image_path = image_entry.path
                image = self._read_image(image_path)
                keypoints, descriptors = self._compute_sift_features(image)

                if dictionary_split:
                    random_samples = self._get_sample_descriptors(
                        list(zip(keypoints, descriptors, strict=True)),
                        dictionary_random_sample_size,
                    )
                    keypoints, descriptors = zip(*random_samples, strict=True)

                sift_features = self._sift_features_to_dataframe(image_path, keypoints, descriptors)
                split_dataframe = pd.concat([split_dataframe, sift_features])

        return split_dataframe

    def create_train_test_split(self, train_image_path: str, test_image_path: str) -> tuple:
        """
        Creates a train-test split for the images in the specified directories.

        Args:
            train_image_path (str): The path to the directory containing the training images.
            test_image_path (str): The path to the directory containing the test images.

        Returns:
            tuple: A tuple containing the train DataFrame and the test DataFrame.
        """
        for subfolder in os.listdir(train_image_path):
            train_per_class = self._create_simple_split(os.path.join(train_image_path, subfolder))
            train_per_class['target'] = subfolder
            self.train = pd.concat([self.train, train_per_class])

        for subfolder in os.listdir(test_image_path):
            test_per_class = self._create_simple_split(os.path.join(test_image_path, subfolder))
            test_per_class['target'] = subfolder
            self.test = pd.concat([self.test, test_per_class])

        return self.train, self.test

    def create_dictionary_split(
        self, dictionary_image_path: str, dictionary_random_sample_size: float
    ) -> pd.DataFrame:
        """
        Creates a dictionary split for the images in the specified directory.

        Args:
            dictionary_image_path (str): The path to the directory containing the dictionary images.
            dictionary_random_sample_size (float): The fraction of randomly selected samples.

        Returns:
            pd.DataFrame: A DataFrame representing the dictionary split.
        """
        for subfolder in os.listdir(dictionary_image_path):
            dict_per_class = self._create_simple_split(
                os.path.join(dictionary_image_path, subfolder), True, dictionary_random_sample_size
            )
            dict_per_class['target'] = subfolder
            self.dictionary = pd.concat([self.dictionary, dict_per_class])
        return self.dictionary
