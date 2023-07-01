from bovw_cam.bovw import BoVW


def test_if_kmeans_is_instantiated_correctly(extracted_features):
    bovw = BoVW(extracted_features, 2, 16)
    assert len(bovw.mini_batch_kmeans.cluster_centers_) == 2


def test_if_histograms_are_calculated_correctly(extracted_features):
    bovw = BoVW(extracted_features, 2, 16)
    single_image_features = extracted_features[extracted_features['image_path'] == 'image_1.jpg']
    histogram, _ = bovw._get_single_histogram(single_image_features[['feature_1', 'feature_2', 'feature_3', 'feature_4']])
    assert histogram in ([1, 0], [0, 1])