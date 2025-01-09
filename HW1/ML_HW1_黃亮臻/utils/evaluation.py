from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def evaluate_performance(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    return metrics


def save_all_results_to_csv(best_results, combined_accuracy, final_metrics, final_predictions, output_file="model_results.csv"):
    """
    保存所有模型的最佳參數、指標結果以及加權投票結果到 CSV 文件。
    
    Args:
        best_results (dict): 各個分類器的最佳結果（包含參數和性能指標）。
        combined_accuracy (float): 加權投票的最終準確率。
        final_metrics (dict): 加權投票的性能指標（如 Accuracy, F1-score）。
        final_predictions (np.ndarray): 加權投票的最終預測結果。
        output_file (str): 保存文件的名稱。
    """
    # Prepare data for each classifier
    data = []
    for name, result in best_results.items():
        row = {
            "Model": name,
            "Best Parameter": result["Parameter"],
            "Accuracy": result["Accuracy"],
            "F1-score": result.get("F1-score", None),  # Assuming F1-score might not always be present
        }
        data.append(row)
    
    # Add the combined (weighted voting) result
    data.append({
        "Model": "Weighted Voting",
        "Best Parameter": "N/A",
        "Accuracy": combined_accuracy,
        "F1-score": final_metrics.get("F1-score", None),
    })

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Save predictions as a separate CSV file
    predictions_df = pd.DataFrame({"Final Predictions": final_predictions})
    predictions_output_file = output_file.replace(".csv", "_predictions.csv")
    predictions_df.to_csv(predictions_output_file, index=False)
    print(f"Predictions saved to {predictions_output_file}")

