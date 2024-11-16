import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# Paths
original_dataset_dir = "archive"
output_dir = "miniImageNet"

# Parameters
image_size = (32, 32)  # Target size for images
train_samples_per_class = 500
test_samples_per_class = 100

# Create output directories
os.makedirs(output_dir, exist_ok=True)
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
print(train_dir)
print(test_dir)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Read the original dataset
image_paths = []
labels = []

for label in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, label)
    if os.path.isdir(class_dir):
        for img_file in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_file))
            labels.append(label)

# Create a DataFrame for easy handling
data = pd.DataFrame({"image_path": image_paths, "label": labels})

# Split into training and test sets
train_data = []
test_data = []

for label in data["label"].unique():
    class_data = data[data["label"] == label]
    train, test = train_test_split(class_data, train_size=train_samples_per_class, test_size=test_samples_per_class, random_state=42)
    train_data.append(train)
    test_data.append(test)

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

# Function to process and save images
def process_and_save_images(data, output_folder):
    processed_data = []
    for _, row in data.iterrows():
        img = Image.open(row["image_path"]).convert("RGB")
        img_resized = img.resize(image_size, Image.ANTIALIAS)
        label = row["label"]
        label_dir = os.path.join(output_folder, label)
        os.makedirs(label_dir, exist_ok=True)
        output_path = os.path.join(label_dir, os.path.basename(row["image_path"]))
        img_resized.save(output_path)
        processed_data.append({"image_path": output_path, "label": label})
    return processed_data

# Process and save training images
train_processed = process_and_save_images(train_data, train_dir)

# Process and save test images
test_processed = process_and_save_images(test_data, test_dir)

# Save labels as CSV
pd.DataFrame(train_processed).to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
pd.DataFrame(test_processed).to_csv(os.path.join(output_dir, "test_labels.csv"), index=False)

print("Dataset preprocessing completed!")
