# Assignment I: Collaborative and Adaptive Classifier Systems


## Data Preparation

1. Download the MiniImageNet dataset from [Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet/data).

2. Prepare Dataset  
   Run the following script to preprocess the dataset:
   ```bash
   python data_preparation.py
   ```

3. Dataset Structure  
   After running the script, the dataset will be organized into the following format:
   - Training Set:  
     - 32x32 RGB images  
     - 100 classes  
     - 500 images per class  
   - Test Set:  
     - 32x32 RGB images  
     - 100 classes  
     - 100 images per class  
   - Label Format:  
     - `image_path`: Path to the image file  
     - `label`: Corresponding class label  

---

## Q1: Collaborative Multi-Classifier System Design

### Run the System
Evaluate multiple classifiers and their performance by running:
```bash
python main1.py --dataset_dir miniImageNet --mode all --knn_k_values 5 10 20 --svm_c_values 0.5 1 10 --dt_max_depth_values 5 10 20
```

### Explanation of Parameters

- `--dataset_dir miniImageNet`  
  Specifies the dataset directory containing `train_labels.csv` and `test_labels.csv`.

- `--mode [auto|select|all]`  
  Determines the operational mode:
  - `auto`: Automatically selects the best classifier based on dataset characteristics (e.g., noise ratio, feature correlation).
  - `select`: Prompts the user to manually select a classifier (`knn`, `svm`, or `dt`).
  - `all`: Runs all classifiers, optimizes their parameters, and performs weighted voting to determine the best predictions.

- `--knn_k_values 5 10 20`  
  Specifies the values of `k` for k-NN during hyperparameter tuning.

- `--svm_c_values 0.5 1 10`  
  Specifies the regularization parameter `C` values for SVM during hyperparameter tuning.

- `--dt_max_depth_values 5 10 20`  
  Specifies the maximum depth values for Decision Tree during hyperparameter tuning.

---

## Q2: Adaptive Classifier Adjustment System

This system dynamically selects the appropriate classifier and parameters based on dataset conditions, such as label skewness (imbalance) and data density.

### Run the Adaptive System

1. Run with Default Imbalance Settings  
   ```bash
   python main2.py --dataset_dir miniImageNet --imbalance --seed 123
   ```

2. Customize Thresholds and Parameters  
   ```bash
   python main2.py --dataset_dir miniImageNet \
                  --skew_threshold 0.7 \
                  --max_skewness 5.0 \
                  --C_min 0.01 \
                  --C_max 300 \
                  --max_density 1000 \
                  --k_min 3 \
                  --k_max 200
   ```

---

#### Run with Default Parameters:
```bash
python main2.py --dataset_dir miniImageNet --imbalance --seed 42
```

#### Custom Thresholds and Parameter Ranges:
```bash
python main2.py --dataset_dir miniImageNet \
                --skew_threshold 0.6 \
                --max_skewness 5.0 \
                --C_min 0.05 \
                --C_max 200 \
                --max_density 500 \
                --k_min 10 \
                --k_max 100
```

### Explanation of Arguments

1. `--dataset_dir` (Required)
   - Specifies the directory containing the dataset files.
   - Example: `miniImageNet`

2. `--imbalance` (Optional)
   - Flag to generate and use an imbalanced dataset based on a random ratio.
   - Usage: Add this flag if you want to create an imbalanced dataset.

3. `--seed` (Optional, Default: `42`)
   - Random seed for reproducibility when generating imbalanced datasets or other random operations.
   - Example: `--seed 123`

4. `--skew_threshold` (Optional, Default: `0.5`)
   - Threshold to determine if the dataset is considered skewed.
   - If the skewness of label distribution is above this value, the system selects SVM.

5. `--max_skewness` (Optional, Default: `3.0`)
   - Maximum skewness value for mapping to SVM's `C` parameter.
   - Higher skewness maps to smaller `C`.

6. `--C_min` (Optional, Default: `0.01`)
   - Minimum value for SVM's regularization parameter `C`.

7. `--C_max` (Optional, Default: `300`)
   - Maximum value for SVM's regularization parameter `C`.

8. `--max_density` (Optional, Default: `1000`)
   - Maximum density value for mapping to k-NN's `k` parameter.
   - Higher density maps to larger `k`.

9. `--k_min` (Optional, Default: `3`)
   - Minimum value for k-NN's `k` parameter.

10. `--k_max` (Optional, Default: `200`)
    - Maximum value for k-NN's `k` parameter.
