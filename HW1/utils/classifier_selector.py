from sklearn.cluster import DBSCAN
import numpy as np

class ClassifierSelector:
    def __init__(self, noise_threshold=0.1, correlation_threshold=0.2, eps=0.2, min_samples=10):
        """
        初始化分類器選擇器。

        Args:
            noise_threshold (float): 噪音比例閾值，低於此值選擇 k-NN。
            correlation_threshold (float): 特徵相關性閾值，高於此值選擇 Decision Tree。
            eps (float): DBSCAN 的鄰域半徑。
            min_samples (int): DBSCAN 的最小鄰域樣本數。
        """
        self.noise_threshold = noise_threshold
        self.correlation_threshold = correlation_threshold
        self.eps = eps
        self.min_samples = min_samples

    @staticmethod
    def calculate_average_correlation(X):
        """
        計算所有特徵之間的相關性，並取平均值。

        Args:
            X (np.ndarray): 特徵數據集，形狀為 (n_samples, n_features)。

        Returns:
            float: 所有特徵兩兩組合的平均相關性。
        """
        n_features = X.shape[1]
        correlations = []

        # 計算所有特徵兩兩組合的相關性
        for i in range(n_features):
            for j in range(i + 1, n_features):
                correlation = np.corrcoef(X[:, i], X[:, j])[0, 1]
                correlations.append(abs(correlation))  # 取絕對值

        # 返回平均相關性
        return np.mean(correlations) if correlations else 0

    def check_noise_with_dbscan(self, X):
        """
        使用 DBSCAN 檢查數據中的噪音比例。

        Args:
            X (np.ndarray): 數據集，形狀為 (n_samples, n_features)。

        Returns:
            float: 噪音點比例。
        """
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean")
        labels = dbscan.fit_predict(X)
        noise_ratio = np.sum(labels == -1) / len(labels)
        return noise_ratio

    def select(self, X):
        """
        根據數據特性自動選擇分類器。

        Args:
            X (np.ndarray): 特徵數據集，形狀為 (n_samples, n_features)。

        Returns:
            str: 選擇的分類器名稱。
            object: 初始化好的分類器實例。
        """
        # Step 1: 檢查噪音
        noise_ratio = self.check_noise_with_dbscan(X)
        print(f"噪音比例: {noise_ratio:.2%}")

        if noise_ratio < self.noise_threshold:
            print("噪音比例較低，選擇 k-NN 分類器")
            return "knn"

        # Step 2: 計算所有特徵的平均相關性
        avg_correlation = self.calculate_average_correlation(X)
        print(f"平均特徵相關係數: {avg_correlation:.2f}")

        if avg_correlation > self.correlation_threshold:
            print("特徵相關性較高，選擇 Decision Tree")
            return "decision tree"

        # Step 3: 噪音較大且相關性低，選擇 SVM
        print("噪音較大且特徵相關性低，選擇 SVM 分類器")
        return "svm"