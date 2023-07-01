from bovw_cam.bovw_cam import BoVWCAM
import numpy as np


def test_if_max_pooling_is_working(classifier, extracted_features, fake_image):
    bovw_cam = BoVWCAM(classifier, extracted_features)
    expected_output = np.array([[5]])
    assert np.array_equal(bovw_cam._max_pooling_2d(fake_image, kernel_size=2, stride=2), expected_output)


def test_if_the_correlation_matrix_was_calculated_correctly(classifier, extracted_features, histograms):
    bovw_cam = BoVWCAM(classifier, extracted_features)
    corr_matrix = bovw_cam._calculate_correlation_matrix(histograms)
    assert corr_matrix == [[0.9999999999999999, -0.9999999999999999], [-0.9999999999999999, 0.9999999999999999]]