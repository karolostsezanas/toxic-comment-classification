# Toxic Comment Classification Project

This VS Code project is designed for the Kaggle Toxic Comment Classification Challenge.

## What you need to do first

1. Download these files from Kaggle and place them in the `data` folder:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

2. Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Check whether the project sees your GPU:

```powershell
python src/check_device.py --device auto
python src/check_device.py --device cpu
python src/check_device.py --device cuda
```

4. Run EDA and generate graphs:

```powershell
python src/01_eda.py
```

5. Test experiments on a smaller sample first:

```powershell
python src/02_train_eval.py --feature tfidf_svd --device cpu --sample 30000
python src/02_train_eval.py --feature sbert --device auto --sample 30000
```

6. Run full experiments when the test works:

```powershell
python src/02_train_eval.py --feature tfidf_svd --device cpu
python src/02_train_eval.py --feature sbert --device auto
```

7. Build a Kaggle submission:

```powershell
python src/03_make_submission.py --feature tfidf_full --model logreg --device cpu
```

## Notes about CPU and GPU

The scikit-learn classifiers in this project run on CPU. The `--device` switch is mainly used by SentenceTransformer embeddings. If you use `--device auto`, the code uses CUDA when PyTorch detects it, otherwise it falls back to CPU.

For the report, compare:
- TF-IDF with SVD features
- SentenceTransformer embeddings

Across the same three models:
- Logistic Regression
- Linear SVM
- Random Forest

Recommended final Kaggle submission:
- TF-IDF full word and character features
- Logistic Regression

This is separated from the comparison experiment because it is usually stronger for leaderboard scoring.
