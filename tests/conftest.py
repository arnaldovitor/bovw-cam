from typing import Any
import cv2
import pandas as pd
import numpy as np
from pytest import fixture
from bovw_cam.classifier_interface import ClassifierInterface


class Classifier(ClassifierInterface):
    def __init__(self, model: Any) -> None:
        super().__init__(model)
    
    def predict(self, input_sample: Any) -> None:
        return None


@fixture
def correct_image_path():
    return './tests/test_images/dog.png'


@fixture
def wrong_image_path():
    return './tests/test_images/wrong.png'


@fixture
def dog_image():
    return cv2.imread('./tests/test_images/dog.png')


@fixture
def extracted_features():
    data = {
        'image_path': ['image_1.jpg', 'image_2.jpg', 'image_3.jpg', 'image_4.jpg', 'image_5.jpg'],
        'keypoint_coord_x': [100, 200, 300, 400, 500],
        'keypoint_coord_y': [50, 75, 100, 125, 150],
        'target': [1, 0, 0, 1, 1],
        'feature_1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'feature_2': [0.5, 0.4, 0.3, 0.2, 0.1],
        'feature_3': [10, 20, 30, 40, 50],
        'feature_4': [5, 10, 15, 20, 25]
    }
    
    return pd.DataFrame(data)


@fixture
def classifier():
    return Classifier(None)


@fixture
def fake_image():
    return  np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])


@fixture
def histograms():
    data = {'histogram_1': [50, 20], 
            'histogram_2': [20, 30],
            'target': [1, 0]}
    
    return pd.DataFrame(data)