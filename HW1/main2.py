import argparse
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from utils.load_data import DataLoader
from utils.calculate import calculate_skewness, calculate_density
from utils.map import map_skew_to_c, map_density_to_k
from utils.evaluation import evaluate_performance


def main(args):
    # Initialize DataLoader
    data_loader = DataLoader(args.dataset_dir)

    # Load original dataset
    try:
        X_train, y_train = data_loader.load_data(f"{args.dataset_dir}/train_labels.csv")
        X_test, y_test = data_loader.load_data(f"{args.dataset_dir}/test_labels.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Handle imbalance if specified
    if args.imbalance:
        print("Generating an imbalanced dataset...")
        X_imbalanced, y_imbalanced = data_loader.create_imbalanced_dataset(X_train, y_train, args.seed)

        # Plot class distribution for the imbalanced dataset
        data_loader.plot_class_distribution(y_imbalanced, title="Imbalanced Dataset Class Distribution", seed=args.seed)

        # Update training data with imbalanced data
        X_train, y_train = X_imbalanced, y_imbalanced

    # Check for empty datasets
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Dataset is empty.")
        return

    # Calculate skewness of the label distribution
    skewness = calculate_skewness(y_train)
    print(f"Label Skewness: {skewness:.2f}")

    # Adaptive classifier selection
    if skewness > args.skew_threshold:
        # Skewed data: Use SVM
        print("Data is skewed. Using SVM.")
        selected_c = map_skew_to_c(skewness, max_skewness=args.max_skewness, C_min=args.C_min, C_max=args.C_max)
        print(f"Selected SVM C: {selected_c:.4f}")
        model = SVC(C=selected_c, kernel="rbf", probability=True)
    else:
        # Balanced data: Use k-NN
        print("Data is balanced. Using k-NN.")
        data_density = calculate_density(X_train)
        selected_k = map_density_to_k(data_density, max_density=args.max_density, k_min=args.k_min, k_max=args.k_max)
        print(f"Selected k-NN k: {selected_k}")
        model = KNeighborsClassifier(n_neighbors=selected_k)

    # Train the selected model
    print("Training the model...")
    model.fit(X_train, y_train)

    # Test the model
    print("Testing the model...")
    y_pred = model.predict(X_test)
    metrics = evaluate_performance(y_test, y_pred)
    print("Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    # Save the model
    model_type = 'svm' if skewness > args.skew_threshold else 'knn'
    model_name = f"model/best_model_{model_type}_seed_{args.seed}.joblib"
    joblib.dump(model, model_name)
    print(f"Model saved to {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Classifier System")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--imbalance", action="store_true", help="Generate and use an imbalanced dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skew_threshold", type=float, default=0.5, help="Threshold to determine if data is skewed")
    parser.add_argument("--max_skewness", type=float, default=3.0, help="Maximum skewness value for mapping")
    parser.add_argument("--C_min", type=float, default=0.01, help="Minimum C value for SVM")
    parser.add_argument("--C_max", type=float, default=300, help="Maximum C value for SVM")
    parser.add_argument("--max_density", type=float, default=1000, help="Maximum density value for mapping")
    parser.add_argument("--k_min", type=int, default=3, help="Minimum k value for k-NN")
    parser.add_argument("--k_max", type=int, default=200, help="Maximum k value for k-NN")
    args = parser.parse_args()

    main(args)
