import numpy as np

def map_skew_to_c(skewness, max_skewness=3.0, C_min=0.01, C_max=300):
    """
    映射不平衡程度到 SVM 的 C 值。
    
    Args:
        skewness (float): 標籤分布的偏態。
        max_skewness (float): 偏態的最大值（假設最大不平衡程度）。
        C_min (float): SVM C 值的最小值。
        C_max (float): SVM C 值的最大值。
    
    Returns:
        float: 動態選擇的 SVM C 值。
    """
    # 對數平滑
    weight = np.log1p(max_skewness - min(skewness, max_skewness)) / np.log1p(max_skewness)
    C = C_min + weight * (C_max - C_min)
    print(f"Selected SVM C: {C:.4f}")
    return C


def map_density_to_k(density, max_density=1000, k_min=3, k_max=200):
    """
    映射密度程度到 k-NN 的 k 值。
    
    Args:
        density (float): 數據集的密度。
        max_density (float): 假設的最大密度。
        k_min (int): k-NN 的最小 k 值。
        k_max (int): k-NN 的最大 k 值。
    
    Returns:
        int: 動態選擇的 k 值。
    """
    # 對數平滑
    weight = np.log1p(density) / np.log1p(max_density)
    weight = max(0, min(weight, 1))  # 確保在 [0, 1] 範圍內
    k = int(k_min + weight * (k_max - k_min))
    print(f"Selected k-NN k: {k}")
    return k
