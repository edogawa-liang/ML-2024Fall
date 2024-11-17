import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import skew


def calculate_density(X, sample_size=1000):

    # Ensure the sample size is not larger than the dataset size
    if len(X) > sample_size:
        X_sample = X[np.random.choice(len(X), sample_size, replace=False)]
    else:
        X_sample = X
    
    # Calculate pairwise distances on the sampled data
    distances = pairwise_distances(X_sample, metric="euclidean")
    
    # Ignore diagonal entries (distance to self)
    np.fill_diagonal(distances, np.nan)
    mean_distance = np.nanmean(distances)
    
    # Density defined as number of samples divided by average distance
    density = len(X_sample) / mean_distance if mean_distance > 0 else len(X_sample)
    print(f"Approximated Data Density: {density:.2f}")
    return density


def calculate_skewness(y):
    """
    計算標籤分佈的偏態（Skewness）。
    
    Args:
        y (np.ndarray): 標籤數據（一維數組，包含每個樣本的標籤）。
    
    Returns:
        float: 標籤分佈的偏態值。
    """
    # 計算每個標籤的頻率分布
    unique, counts = np.unique(y, return_counts=True)
    
    # 計算頻率的偏態值
    skewness = skew(counts)
    print(f"Label Skewness: {skewness:.2f}")
    return np.abs(skewness)
