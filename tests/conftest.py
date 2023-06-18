import cv2
from pytest import fixture


@fixture
def correct_image_path():
    return './tests/test_images/dog.png'


@fixture
def wrong_image_path():
    return './tests/test_images/wrong.png'


@fixture
def dog_image():
    return cv2.imread('./tests/test_images/dog.png')
