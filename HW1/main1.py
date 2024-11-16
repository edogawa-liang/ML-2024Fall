import argparse
import numpy as np
from utils.classifier_runner import run_selected_classifier
from utils.classifier_selector import ClassifierSelector
from utils.experiment import ExperimentRunner
from utils.evaluation import evaluate_performance, save_all_results_to_csv


def parse_arguments():
    parser = argparse.ArgumentParser(description="Classifier Selection and Evaluation Script")
    
    # Dataset directory
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset")
    
    # Parameters for each classifier
    parser.add_argument("--knn_k_values", type=int, nargs="+", default=[3, 5, 10], help="k values for k-NN classifier")
    parser.add_argument("--svm_c_values", type=float, nargs="+", default=[0.5, 1, 5], help="C values for SVM classifier")
    parser.add_argument("--dt_max_depth_values", type=int, nargs="+", default=[10, 15, 20],
                        help="Max depth values for Decision Tree classifier")

    # Auto mode parameters
    parser.add_argument("--noise_threshold", type=float, default=0.1, help="Noise threshold for auto mode")
    parser.add_argument("--correlation_threshold", type=float, default=0.2, help="Correlation threshold for auto mode")
    parser.add_argument("--eps", type=float, default=0.2, help="Epsilon value for DBSCAN in auto mode")
    parser.add_argument("--min_samples", type=int, default=10, help="Minimum samples for DBSCAN in auto mode")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True, choices=["auto", "select", "all"], help="Mode: auto, select, or all")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Initialize runner and load data
    runner = ExperimentRunner(args.dataset_dir)
    X_train, y_train, X_test, y_test = runner.load_data()

    if args.mode == "auto":
        # Auto mode: Use ClassifierSelector
        print("Running auto mode...")
        selector = ClassifierSelector(noise_threshold=args.noise_threshold, 
                                       correlation_threshold=args.correlation_threshold, 
                                       eps=args.eps, 
                                       min_samples=args.min_samples)
        classifier_name = selector.select(X_train)
        print(f"Auto-selected classifier: {classifier_name}")
        best_result = run_selected_classifier(classifier_name, runner, args.knn_k_values, args.svm_c_values, args.dt_max_depth_values, X_train, y_train, X_test, y_test)

    elif args.mode == "select":
        # Manual mode: Select a classifier
        classifier_name = input("Select classifier: [knn/svm/dt]: ").strip().lower()
        best_result = run_selected_classifier(classifier_name, runner, args.knn_k_values, args.svm_c_values, args.dt_max_depth_values, X_train, y_train, X_test, y_test)
        y_pred = best_result["Model"].predict(X_test)
        print("Performance Metrics:")
        metrics = evaluate_performance(y_test, y_pred)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

    elif args.mode == "all":
        # All mode: Evaluate all classifiers and calculate weighted voting
        best_knn = runner.find_best_knn(args.knn_k_values, X_train, y_train, X_test, y_test)
        best_svm = runner.find_best_svm(args.svm_c_values, X_train, y_train, X_test, y_test)
        best_dt = runner.find_best_decision_tree(args.dt_max_depth_values, X_train, y_train, X_test, y_test)
        best_results = {"k-NN": best_knn, "SVM": best_svm, "Decision Tree": best_dt}
        combined_accuracy = runner.calculate_weighted_voting(best_results, X_test, y_test)
        print(f"Weighted Voting Accuracy: {combined_accuracy:.2f}")

        for name, result in best_results.items():
            y_pred = result["Model"].predict(X_test)
            print(f"Performance Metrics for {name}:")
            metrics = evaluate_performance(y_test, y_pred)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")

        print("Performance Metrics for Weighted Voting System:")
        final_preds = np.argmax(
            sum(result["Accuracy"] * result["Probs"] for result in best_results.values() if result["Probs"] is not None),
            axis=1
        )
        metrics = evaluate_performance(y_test, final_preds)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

        save_all_results_to_csv(best_results, combined_accuracy, metrics, final_preds)
