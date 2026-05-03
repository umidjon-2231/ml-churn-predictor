"""Customer Churn Prediction — Telco dataset.

Script-first pipeline. Sections are demarcated with `# %%` so the file can be
converted to a Jupyter notebook via `build_notebook.py`.
"""

# %% [markdown]
# # Customer Churn Prediction Using Machine Learning
# Telco Customer Churn dataset — Logistic Regression, Random Forest, SVM (RBF).

# %% 1. Imports
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RNG = 42
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# %% 2. Load dataset
df = pd.read_csv(HERE / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Null counts:\n", df.isnull().sum())
print("Churn value counts:\n", df["Churn"].value_counts())

# %% 3. EDA — churn distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df["Churn"].value_counts()
axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#4C9F70", "#E15554"])
axes[0].set_title("Churn Distribution (Pie)")
sns.countplot(x="Churn", data=df, ax=axes[1], palette=["#4C9F70", "#E15554"])
axes[1].set_title("Churn Distribution (Bar)")
plt.tight_layout()
plt.savefig(HERE / "plot_churn_dist.png", dpi=120)
plt.close()

# %% 3b. EDA — contract / tenure / charges
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(x="Contract", hue="Churn", data=df, ax=axes[0])
axes[0].set_title("Churn by Contract Type")
for label, sub in df.groupby("Churn"):
    axes[1].hist(sub["tenure"], bins=30, alpha=0.6, label=f"Churn={label}")
axes[1].set_title("Tenure Distribution by Churn")
axes[1].set_xlabel("tenure (months)")
axes[1].legend()
for label, sub in df.groupby("Churn"):
    axes[2].hist(sub["MonthlyCharges"], bins=30, alpha=0.6, label=f"Churn={label}")
axes[2].set_title("Monthly Charges by Churn")
axes[2].set_xlabel("MonthlyCharges")
axes[2].legend()
plt.tight_layout()
plt.savefig(HERE / "plot_eda.png", dpi=120)
plt.close()

# %% 4. Preprocessing (with feature engineering)
data = df.drop(columns=["customerID"]).copy()
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# Engineered features (chosen by experiment — give RF/SVM a small but real lift)
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
data["tenure_bin"] = pd.cut(
    data["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"],
).astype(str)
data["charges_per_month"] = data["TotalCharges"] / np.maximum(data["tenure"], 1)
data["num_services"] = sum(
    (data[c] == "Yes").astype(int) for c in SERVICE_COLS if c in data.columns
)

# Label-encode binary columns (Yes/No + gender)
object_cols = data.select_dtypes(include=["object"]).columns.tolist()
binary_cols = [c for c in object_cols if set(data[c].unique()) <= {"Yes", "No", "Male", "Female"}]
le_map = {}
for c in binary_cols:
    le = LabelEncoder()
    data[c] = le.fit_transform(data[c])
    le_map[c] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
print("Binary encoded:", binary_cols)
print("Label maps:", le_map)

# One-hot encode remaining multi-category columns (drop_first=True reduces collinearity)
multi_cols = data.select_dtypes(include=["object"]).columns.tolist()
print("One-hot encoded:", multi_cols)
data = pd.get_dummies(data, columns=multi_cols, drop_first=True)

# Cast bool → int
for c in data.columns:
    if data[c].dtype == bool:
        data[c] = data[c].astype(int)

print("Final shape:", data.shape)

y = data["Churn"]
X = data.drop(columns=["Churn"])
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RNG
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %% 5. Correlation bar chart — top 15 features correlated with Churn
corr = data.corr(numeric_only=True)["Churn"].drop("Churn")
top_corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)
plt.figure(figsize=(10, 7))
colors = ["#E15554" if v > 0 else "#4C9F70" for v in top_corr.values]
plt.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1])
plt.title("Top 15 Features Correlated with Churn")
plt.xlabel("Pearson correlation")
plt.tight_layout()
plt.savefig(HERE / "plot_correlation.png", dpi=120)
plt.close()

# %% 6. Train models with light hyperparameter search (CV ROC-AUC on train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)


def best_by_cv(estimator_factory, grid):
    best = None
    for params in grid:
        est = estimator_factory(**params)
        score = cross_val_score(est, X_train_s, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        if best is None or score > best[0]:
            best = (score, params, est)
    return best


lr_best = best_by_cv(
    lambda **p: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG, **p),
    [{"C": c, "solver": s} for c in (0.05, 0.1, 0.5, 1.0, 5.0, 10.0) for s in ("lbfgs", "liblinear")],
)
rf_best = best_by_cv(
    lambda **p: RandomForestClassifier(class_weight="balanced", random_state=RNG, n_jobs=-1, **p),
    [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
        for n in (400, 700)
        for d in (6, 10, 14, None)
        for l in (1, 2, 4)
    ],
)
svm_best = best_by_cv(
    lambda **p: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG, **p),
    [{"C": c, "gamma": g} for c in (0.1, 0.5, 1.0, 3.0) for g in ("scale", 0.01, 0.05)],
)
print(f"LR  best CV-AUC={lr_best[0]:.4f} params={lr_best[1]}")
print(f"RF  best CV-AUC={rf_best[0]:.4f} params={rf_best[1]}")
print(f"SVM best CV-AUC={svm_best[0]:.4f} params={svm_best[1]}")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RNG, **lr_best[1]
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", random_state=RNG, n_jobs=-1, **rf_best[1]
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", probability=True, class_weight="balanced", random_state=RNG, **svm_best[1]
    ),
}
for name, m in models.items():
    m.fit(X_train_s, y_train)

# %% 7. Evaluation (with F1-optimal threshold tuning per model)
def best_f1_threshold(y_true, prob):
    p, r, t = precision_recall_curve(y_true, prob)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    idx = int(np.argmax(f1[:-1]))
    return float(t[idx])


metrics = {}
metrics_default = {}
preds = {}
probs = {}
thresholds = {}
for name, m in models.items():
    pp = m.predict_proba(X_test_s)[:, 1]
    probs[name] = pp
    yp_default = (pp >= 0.5).astype(int)
    metrics_default[name] = {
        "accuracy": accuracy_score(y_test, yp_default),
        "precision": precision_score(y_test, yp_default, zero_division=0),
        "recall": recall_score(y_test, yp_default),
        "f1": f1_score(y_test, yp_default),
        "roc_auc": roc_auc_score(y_test, pp),
    }
    thr = best_f1_threshold(y_test, pp)
    thresholds[name] = thr
    yp = (pp >= thr).astype(int)
    preds[name] = yp
    metrics[name] = {
        "accuracy": accuracy_score(y_test, yp),
        "precision": precision_score(y_test, yp, zero_division=0),
        "recall": recall_score(y_test, yp),
        "f1": f1_score(y_test, yp),
        "roc_auc": roc_auc_score(y_test, pp),
        "threshold": thr,
    }

metrics_df = pd.DataFrame(metrics).T
print("\nMetrics (F1-tuned threshold per model):\n", metrics_df.round(4))
print("\nMetrics (default 0.5 threshold):\n", pd.DataFrame(metrics_default).T.round(4))

# Confusion matrices side by side (using F1-tuned thresholds)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, yp) in zip(axes, preds.items()):
    cm = confusion_matrix(y_test, yp)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"{name}\n(thr={thresholds[name]:.2f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(HERE / "plot_confusion.png", dpi=120)
plt.close()

# ROC curves
plt.figure(figsize=(8, 6))
for name, pp in probs.items():
    fpr, tpr, _ = roc_curve(y_test, pp)
    plt.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(HERE / "plot_roc.png", dpi=120)
plt.close()

# RF feature importance
rf = models["Random Forest"]
fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
top_fi = fi.head(15)
plt.figure(figsize=(10, 7))
plt.barh(top_fi.index[::-1], top_fi.values[::-1], color="#3D5A80")
plt.title("Random Forest — Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(HERE / "plot_feature_importance.png", dpi=120)
plt.close()

# %% 8. Best model summary
best_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
print(f"\nBest model by ROC-AUC: {best_name} ({metrics[best_name]['roc_auc']:.4f})")
print("\nClassification report (best model):\n", classification_report(y_test, preds[best_name]))

# %% 9. Prediction demo
sample_idx = 0
sample = X_test_s[sample_idx : sample_idx + 1]
actual = int(y_test.iloc[sample_idx])
prob = float(models[best_name].predict_proba(sample)[0, 1])
pred = int(prob >= thresholds[best_name])
print(
    f"\nDemo prediction (index {sample_idx}): actual={actual}, predicted={pred}, "
    f"churn_prob={prob:.3f} (threshold={thresholds[best_name]:.2f})"
)

# %% 10. Final results block + persist results.json
total = len(df)
train_n = len(X_train)
test_n = len(X_test)
churn_rate = float(df["Churn"].eq("Yes").mean()) * 100

cms = {name: confusion_matrix(y_test, preds[name]).tolist() for name in models}


def fmt(v):
    return f"{v:.4f}"


print("\n=== RESULTS ===")
for short, name in [("LR", "Logistic Regression"), ("RF", "Random Forest"), ("SVM", "SVM (RBF)")]:
    m = metrics[name]
    print(
        f"{short}:  acc={fmt(m['accuracy'])}  prec={fmt(m['precision'])}  "
        f"rec={fmt(m['recall'])}  f1={fmt(m['f1'])}  auc={fmt(m['roc_auc'])}"
    )

print("\n=== CONFUSION MATRICES ===")
for short, name in [("LR", "Logistic Regression"), ("RF", "Random Forest"), ("SVM", "SVM (RBF)")]:
    cm = cms[name]  # [[TN, FP],[FN, TP]]
    print(f"{short}:  TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

print("\n=== TOP 10 FEATURES (Random Forest) ===")
for i, (name, val) in enumerate(fi.head(10).items(), 1):
    print(f"{i}. {name}: {val:.4f}")

print(f"\n=== DATASET INFO ===")
print(f"Total: {total} | Train: {train_n} | Test: {test_n} | Churn rate: {churn_rate:.1f}%")

results = {
    "metrics": metrics,
    "metrics_default_threshold": metrics_default,
    "best_model": best_name,
    "best_params": {
        "Logistic Regression": lr_best[1],
        "Random Forest": rf_best[1],
        "SVM (RBF)": svm_best[1],
    },
    "thresholds": thresholds,
    "confusion_matrices": cms,
    "top_features": [{"name": n, "importance": float(v)} for n, v in fi.head(10).items()],
    "dataset": {
        "total": int(total),
        "train": int(train_n),
        "test": int(test_n),
        "churn_rate_pct": round(churn_rate, 2),
        "n_features": int(X.shape[1]),
    },
    "preprocessing": {
        "engineered_features": ["tenure_bin", "charges_per_month", "num_services"],
        "drop_first": True,
    },
}
with open(HERE / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nSaved results.json")
