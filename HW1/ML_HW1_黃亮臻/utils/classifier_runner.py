def run_selected_classifier(classifier_name, runner, knn_k_values, svm_c_values, dt_params_list, X_train, y_train, X_test, y_test):
    """
    根據分類器名稱執行對應的超參數調整並返回結果。

    Args:
        classifier_name (str): `knn`, `svm`, or `dt`
        runner (ExperimentRunner): 用於運行實驗的實例。
        knn_k_values (list): KNN 的超參數範圍。
        svm_c_values (list): SVM 的超參數範圍。
        dt_params_list (list): Decision Tree 的參數範圍。
        X_train (np.ndarray): 訓練數據特徵。
        y_train (np.ndarray): 訓練數據標籤。
        X_test (np.ndarray): 測試數據特徵。
        y_test (np.ndarray): 測試數據標籤。

    Returns:
        dict: 超參數調整結果。
    """
    if classifier_name == "knn":
        best_result = runner.find_best_knn(knn_k_values, X_train, y_train, X_test, y_test)
    elif classifier_name == "dt":
        best_result = runner.find_best_decision_tree(dt_params_list, X_train, y_train, X_test, y_test)
    elif classifier_name == "svm":
        best_result = runner.find_best_svm(svm_c_values, X_train, y_train, X_test, y_test)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    print(f"Best result for {classifier_name}: {best_result}")
    return best_result