import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class DataLoader:
    def __init__(self, dataset_dir, resize=(32, 32)):
        self.dataset_dir = dataset_dir
        self.resize = resize

    def load_data(self, csv_path, random_seed=None):
        """
        Load data from the given CSV file and optionally create an imbalanced dataset with random class sizes.

        Args:
            csv_path (str): Path to the CSV file.
            random_seed (int): Random seed for reproducibility. If None, no imbalance is introduced.

        Returns:
            tuple: Arrays of features (X) and labels (y).
        """
        data = pd.read_csv(csv_path)
        images = []
        labels = []

        for _, row in data.iterrows():
            img = plt.imread(row["image_path"])
            images.append(img.flatten())  # Flatten image to 1D array
            labels.append(row["label"])

        X = np.array(images)
        y = np.array(labels)

        if random_seed is not None:
            return self.create_imbalanced_dataset(X, y, random_seed)

        return X, y

    def create_imbalanced_dataset(self, X, y, seed):
        """
        Create an imbalanced dataset with random class sizes.

        Args:
            X (np.ndarray): Feature array.
            y (np.ndarray): Label array.
            seedd (int): Random seed for reproducibility.

        Returns:
            tuple: Arrays of imbalanced features (X) and labels (y).
        """
        unique_classes = np.unique(y)
        np.random.seed(seed)

        # Generate random sample counts for each class
        class_counts = {class_label: np.random.randint(1, len(np.where(y == class_label)[0]) + 1)
                        for class_label in unique_classes}

        imbalanced_X = []
        imbalanced_y = []

        for class_label, count in class_counts.items():
            class_indices = np.where(y == class_label)[0]
            sampled_indices = np.random.choice(class_indices, count, replace=False)

            imbalanced_X.append(X[sampled_indices])
            imbalanced_y.append(y[sampled_indices])

        print("Random Class Sizes:", class_counts)
        return np.vstack(imbalanced_X), np.hstack(imbalanced_y)

    def plot_class_distribution(self, y, title="Class Distribution", seed=None):
        """
        Plot the class distribution as a bar chart.

        Args:
            y (np.ndarray): Label array.
            title (str): Title of the plot.
            seed (int): Random seed used for creating the imbalance.
        """
        class_counts = Counter(y)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        plt.figure(figsize=(12, 6))
        plt.bar(classes, counts, color="skyblue")
        plt.xlabel("Class Labels")
        plt.ylabel("Number of Samples")
        plt.title(title)
        plt.xticks(classes, rotation=90)
        plt.tight_layout()

        filename = f"class_distribution_seed_{seed}.png" if seed is not None else "class_distribution.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}.")
