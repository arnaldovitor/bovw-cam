import cv2
import numpy as np
from src.feature_extractor import FeatureExtractor


def test_if_the_image_was_loaded_correctly(correct_image_path):
    feature_extractor = FeatureExtractor()
    loaded_image = feature_extractor._read_image(correct_image_path)
    assert type(loaded_image) == np.ndarray


def test_if_loading_an_image_with_wrong_path_throws_exception(wrong_image_path):
    exception_is_raised = False
    feature_extractor = FeatureExtractor()

    try:
        _ = feature_extractor._read_image(wrong_image_path)
    except:
        exception_is_raised = True

    assert exception_is_raised == True


def test_if_the_sift_descriptor_is_in_the_correct_form(dog_image):
    feature_extractor = FeatureExtractor()
    _, descriptors = feature_extractor._compute_sift_features(dog_image)
    assert descriptors[0].shape == (128, )


def test_if_any_keypoint_was_found(dog_image):
    feature_extractor = FeatureExtractor()
    keypoints, _ = feature_extractor._compute_sift_features(dog_image)
    assert type(keypoints[0]) == cv2.KeyPoint