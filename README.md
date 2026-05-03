# Customer Churn Prediction Using Machine Learning

University AIML201 group project. Predicts whether a Telco customer will churn (Yes/No) using three classical models trained on the IBM Telco Customer Churn dataset.

## Team

- **Umidjon Tojiboyev** — code
- **Sardor Fatxullayev** — report
- **Zoirxon Fozilxonov** — slides

## Results

Test set, 80/20 stratified split, 5-fold CV-tuned hyperparameters, F1-optimal decision threshold per model.

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|---------:|----------:|-------:|-------:|--------:|
| Logistic Regression | 0.7814   | 0.5714    | 0.7059 | 0.6316 | 0.8411  |
| **Random Forest**   | 0.7679   | 0.5439    | 0.7781 | **0.6403** | **0.8438** |
| SVM (RBF)           | 0.7764   | 0.5629    | 0.7059 | 0.6263 | 0.8307  |

**Best model: Random Forest** (`n_estimators=700, max_depth=10, min_samples_leaf=4`, `class_weight='balanced'`, `random_state=42`).

Top-5 churn drivers (Random Forest feature importance): `Contract_Two year`, `tenure`, `TotalCharges`, `MonthlyCharges`, `charges_per_month`.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter nbformat python-docx
```

Tested on Python 3.13.

## How to run

```bash
# 1. Run the full pipeline (data prep, training, evaluation, plots, results.json)
python churn_pipeline.py

# 2. Build the Jupyter notebook from the script and execute it
python build_notebook.py
jupyter nbconvert --to notebook --execute churn_prediction.ipynb --inplace

# 3. Build the Word report (uses results.json + project_report_template.docx)
python build_report.py
```

After step 1 you will see the `=== RESULTS ===` block printed to stdout and six `plot_*.png` files plus `results.json` in the project folder.

## Files

| File                                      | Purpose                                                |
|-------------------------------------------|--------------------------------------------------------|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv`    | Source dataset (7043 customers, 21 columns)            |
| `churn_pipeline.py`                       | Full ML pipeline as a script with `# %%` cell markers  |
| `build_notebook.py`                       | Converts the script into `churn_prediction.ipynb`      |
| `churn_prediction.ipynb`                  | Executed notebook with embedded outputs                |
| `experiment.py`                           | Hyperparameter and feature-engineering experiments     |
| `build_report.py`                         | Fills the report template with results from `results.json` |
| `project_report_template.docx`            | University-supplied report template (do not edit)      |
| `churn_report.docx`                       | Final report (auto-generated)                          |
| `results.json`                            | Persisted metrics, params, thresholds, top features    |
| `plot_churn_dist.png`                     | Churn class distribution (pie + bar)                   |
| `plot_eda.png`                            | Churn by contract type, tenure, monthly charges        |
| `plot_correlation.png`                    | Top 15 features by correlation with churn              |
| `plot_confusion.png`                      | Confusion matrices for the three models                |
| `plot_roc.png`                            | ROC curves for the three models                        |
| `plot_feature_importance.png`             | Random Forest top 15 feature importances               |

## Pipeline overview

1. **Load** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7043 rows, 21 columns).
2. **EDA** — churn distribution, churn vs contract type, tenure histograms by churn, monthly-charges histograms by churn.
3. **Preprocessing**
   - drop `customerID`
   - coerce `TotalCharges` to numeric, fill blanks with the column median
   - engineer 3 features: `tenure_bin`, `charges_per_month`, `num_services`
   - label-encode binary columns (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`)
   - one-hot encode multi-category columns with `drop_first=True`
   - cast bool dummies to int
   - 80/20 stratified split, `random_state=42`
   - `StandardScaler` fit on train, applied to test
4. **Train** Logistic Regression, Random Forest, SVM (RBF) — all with `class_weight='balanced'`, `random_state=42`. Hyperparameters chosen by 5-fold CV ROC-AUC on the training set.
5. **Evaluate** at default 0.5 threshold and at per-model F1-optimal threshold; record accuracy / precision / recall / F1 / ROC-AUC and confusion matrices.
6. **Visualise** correlations, confusion matrices, ROC curves, RF feature importances.
7. **Persist** all artefacts to `results.json` for the report builder.

## Reproducibility

Everything is seeded with `random_state=42` (train/test split, all three classifiers, K-fold). Re-running `python churn_pipeline.py` will reproduce the metrics in the table above.
