# Toxic Comment Classification Project

This project implements a machine learning workflow for the Kaggle Toxic Comment Classification Challenge. It includes exploratory data analysis, model training and evaluation, combined result comparison, and Kaggle submission generation.

## Project structure

```text
toxic_comment_project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── outputs/
│   ├── figures/
│   ├── results/
│   └── submissions/
├── src/
│   ├── eda.py
│   ├── train_eval.py
│   ├── compare_results.py
│   ├── make_submission.py
│   ├── check_device.py
│   ├── config.py
│   ├── device.py
│   └── text_utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

Download the required files from the Kaggle Toxic Comment Classification Challenge data page and place them inside the `data/` folder:

```text
train.csv
test.csv
sample_submission.csv
```

The optional `test_labels.csv` file is not required for the main workflow.

## Environment setup

Create and activate a Python virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Check CPU or GPU availability

The project includes a device checker. CUDA is mainly relevant for the SBERT embedding stage.

```powershell
python src/check_device.py --device auto
```

You can also explicitly test CPU or CUDA:

```powershell
python src/check_device.py --device cpu
python src/check_device.py --device cuda
```

## 1. Exploratory data analysis

Run:

```powershell
python src/eda.py
```

This generates dataset summaries and visualisations in:

```text
outputs/figures/
outputs/results/
```

Important EDA outputs include:

```text
class_balance.png
labels_per_comment.png
token_length_distribution.png
label_correlation_heatmap.png
top_words_by_label.csv
```

## 2. Train and evaluate models

The training script compares three classifiers:

```text
Logistic Regression
Linear SVM
Random Forest
```

It tests two feature extraction strategies:

```text
TF-IDF with Truncated SVD
SBERT sentence embeddings
```

Run the full experiments:

```powershell
python src/train_eval.py --feature tfidf_svd --device cuda
python src/train_eval.py --feature sbert --device cuda
```

If CUDA is unavailable, use:

```powershell
python src/train_eval.py --feature tfidf_svd --device cpu
python src/train_eval.py --feature sbert --device auto
```

To test faster on a smaller sample:

```powershell
python src/train_eval.py --feature tfidf_svd --device cpu --sample 30000
python src/train_eval.py --feature sbert --device auto --sample 30000
```

The script saves:

```text
per_label_metrics_*.csv
summary_metrics_*.csv
model_*.joblib
roc_*.png
summary_*.png
run_config_*.json
```

## 3. Compare full-dataset results

After running both full experiments, use `compare_results.py` to combine the model summaries into one final comparison table and create combined comparison figures.

Run:

```powershell
python src/compare_results.py
```

This reads the `summary_metrics_*.csv` files from:

```text
outputs/results/
```

It then keeps the full-dataset rows and creates:

```text
outputs/results/combined_model_comparison_full_dataset.csv
```

It also creates the following figures:

```text
outputs/figures/full_dataset_comparison_macro_auc.png
outputs/figures/full_dataset_comparison_macro_f1.png
outputs/figures/full_dataset_comparison_macro_precision.png
outputs/figures/full_dataset_comparison_macro_recall.png
outputs/figures/full_dataset_comparison_hamming_accuracy.png
```

These figures are useful for the final report because they compare all tested model and feature combinations in one place.

## 4. Generate Kaggle submissions

Use `make_submission.py` to train a final model on the full training set and generate a Kaggle-compatible CSV file.

Recommended TF-IDF submission:

```powershell
python src/make_submission.py --feature tfidf_full --model logreg --device cpu
```

SBERT submission:

```powershell
python src/make_submission.py --feature sbert --model logreg --device cuda
```

Generated submission files are saved in:

```text
outputs/submissions/
```

Upload the generated `.csv` files to Kaggle and compare the public and private scores.

## Final selected model

In the completed experiment, full TF-IDF word and character n-gram features with One-vs-Rest Logistic Regression achieved the strongest Kaggle result.

The SBERT Logistic Regression model performed strongest in local validation by macro AUC, but the full TF-IDF Logistic Regression model achieved the best Kaggle submission score.


