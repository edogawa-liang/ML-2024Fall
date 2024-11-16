### Machine Learning Assignment I: Classifier Systems

---

## Data Preparation

1. **Download the MiniImageNet Dataset**  
   [Link to Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet/data)

2. **Prepare the Dataset**  
   Run the preprocessing script:
   ```bash
   python data_preparation.py
   ```
   Dataset structure after preparation:
   - **Training Set**: 100 classes, 500 images per class  
   - **Test Set**: 100 classes, 100 images per class  
   - **Labels**: CSV file with `image_path` and `label`

---

## Q1: Collaborative Classifier System

### Run the System
```bash
python main1.py --dataset_dir miniImageNet --mode all --knn_k_values 5 10 20 --svm_c_values 0.5 1 10 --dt_max_depth_values 5 10 20
```

### Key Parameters
- `--dataset_dir miniImageNet`: Path to the dataset.  
- `--mode`: Operational modes:  
  - `auto`: Automatically selects the best classifier.  
  - `select`: Manually selects a classifier (`knn`, `svm`, or `dt`).  
  - `all`: Runs all classifiers and performs weighted voting.  
- `--knn_k_values 5 10 20`: Values for k-NN's `k` during tuning.  
- `--svm_c_values 0.5 1 10`: Values for SVM's `C` during tuning.  
- `--dt_max_depth_values 5 10 20`: Values for Decision Tree's max depth during tuning.  

---

## Q2: Adaptive Classifier System

This system dynamically selects the best classifier and parameters based on data characteristics (e.g., skewness or density).

### Run the Adaptive System
1. **Default Settings**
   ```bash
   python main2.py --dataset_dir miniImageNet --imbalance --seed 42
   ```

2. **Custom Parameters**
   ```bash
   python main2.py --dataset_dir miniImageNet \
                   --imbalance --seed 42 \
                   --skew_threshold 0.7 \
                   --C_min 0.01 --C_max 100 \
                   --k_min 5 --k_max 50
   ```

### Key Parameters
- `--dataset_dir miniImageNet`: Path to the dataset.  
- `--imbalance`: Generate a random imbalanced dataset.  
- `--seed`: Random seed for reproducibility.  
- `--skew_threshold`: Threshold to decide if data is skewed.  
- `--C_min`, `--C_max`: Range for SVM's `C` parameter.  
- `--k_min`, `--k_max`: Range for k-NN's `k` parameter.  

---