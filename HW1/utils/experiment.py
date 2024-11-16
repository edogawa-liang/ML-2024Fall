import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from utils.load_data import DataLoader
import os
import joblib


class ExperimentRunner:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_loader = DataLoader(dataset_dir)

    def load_data(self):
        X_train, y_train = self.data_loader.load_data(os.path.join(self.dataset_dir, "train_labels.csv"))
        X_test, y_test = self.data_loader.load_data(os.path.join(self.dataset_dir, "test_labels.csv"))

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        return X_train, y_train_encoded, X_test, y_test_encoded

    def find_best_knn(self, knn_k_values, X_train, y_train, X_test, y_test):
        best_result = {"Parameter": None, "Accuracy": 0, "Probs": None, "Model": None}
        for k in knn_k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)

            knn_preds = knn.predict(X_test)
            knn_probs = knn.predict_proba(X_test)
            knn_acc = accuracy_score(y_test, knn_preds)

            if knn_acc > best_result["Accuracy"]:
                best_result.update({
                    "Parameter": f"k={k}",
                    "Accuracy": knn_acc,
                    "Probs": knn_probs,
                    "Model": knn
                })
        print(f"Best k-NN: {best_result}")
        return best_result

    def find_best_svm(self, svm_c_values, X_train, y_train, X_test, y_test):
        best_result = {"Parameter": None, "Accuracy": 0, "Probs": None, "Model": None}
        for c in svm_c_values:
            svm = SVC(C=c, kernel="rbf", probability=True)
            svm.fit(X_train, y_train)

            svm_preds = svm.predict(X_test)
            svm_probs = svm.predict_proba(X_test)
            svm_acc = accuracy_score(y_test, svm_preds)

            if svm_acc > best_result["Accuracy"]:
                best_result.update({
                    "Parameter": f"C={c}",
                    "Accuracy": svm_acc,
                    "Probs": svm_probs,
                    "Model": svm
                })
        print(f"Best SVM: {best_result}")
        return best_result

    def find_best_decision_tree(self, dt_max_depth_values, X_train, y_train, X_test, y_test):
        best_result = {"Parameter": None, "Accuracy": 0, "Probs": None, "Model": None}
        for depth in dt_max_depth_values:
            dt = DecisionTreeClassifier(max_depth=depth)
            dt.fit(X_train, y_train)

            dt_preds = dt.predict(X_test)
            if hasattr(dt, "predict_proba"):
                dt_probs = dt.predict_proba(X_test)
            else:
                dt_probs = None
            dt_acc = accuracy_score(y_test, dt_preds)

            if dt_acc > best_result["Accuracy"]:
                best_result.update({
                    "Parameter": f"max_depth={depth}",
                    "Accuracy": dt_acc,
                    "Probs": dt_probs,
                    "Model": dt
                })
        print(f"Best Decision Tree: {best_result}")
        return best_result

    def save_model(self, model, filename):
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")

    def calculate_weighted_voting(self, best_results, X_test, y_test):
        total_f1 = sum(result["Accuracy"] for result in best_results.values())
        weights = {name: result["Accuracy"] / total_f1 for name, result in best_results.items()}

        combined_probs = np.zeros((len(X_test), len(np.unique(y_test))))
        for name, result in best_results.items():
            combined_probs += weights[name] * result["Probs"]

        final_preds = np.argmax(combined_probs, axis=1)
        combined_acc = accuracy_score(y_test, final_preds)
        print(f"Weighted Voting Accuracy: {combined_acc:.2f}")
        return combined_acc
