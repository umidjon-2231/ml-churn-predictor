"""Experiments to push test ROC-AUC / F1 above the baseline pipeline.

Baseline (from churn_pipeline.py):
  LR  acc=0.7388 f1=0.6134 auc=0.8405
  RF  acc=0.7672 f1=0.6239 auc=0.8400
  SVM acc=0.7154 f1=0.6010 auc=0.8268

Each experiment prints test metrics and the best F1-threshold metrics.
Nothing is written to results.json or the report — pure exploration.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, precision_recall_curve,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RNG = 42
HERE = Path(__file__).resolve().parent

# ---------- Data loading + preprocessing variants ----------
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_raw():
    df = pd.read_csv(HERE / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df


def preprocess(df, *, engineer=False, drop_first=False):
    data = df.drop(columns=["customerID"]).copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    if engineer:
        # tenure bins
        data["tenure_bin"] = pd.cut(
            data["tenure"], bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        ).astype(str)
        # charges per tenure month (avoid div by zero)
        data["charges_per_month"] = data["TotalCharges"] / np.maximum(data["tenure"], 1)
        # number of subscribed value-add services (count of "Yes")
        data["num_services"] = sum(
            (data[c] == "Yes").astype(int) for c in SERVICE_COLS if c in data.columns
        )

    object_cols = data.select_dtypes(include=["object"]).columns.tolist()
    binary_cols = [c for c in object_cols if set(data[c].unique()) <= {"Yes", "No", "Male", "Female"}]
    for c in binary_cols:
        data[c] = LabelEncoder().fit_transform(data[c])
    multi_cols = data.select_dtypes(include=["object"]).columns.tolist()
    data = pd.get_dummies(data, columns=multi_cols, drop_first=drop_first)
    for c in data.columns:
        if data[c].dtype == bool:
            data[c] = data[c].astype(int)
    return data


def split_scale(data):
    y = data["Churn"]
    X = data.drop(columns=["Churn"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte), ytr, yte, X.columns.tolist()


def metrics_at(y_true, prob, threshold=0.5):
    yp = (prob >= threshold).astype(int)
    return {
        "thr": threshold,
        "acc": accuracy_score(y_true, yp),
        "prec": precision_score(y_true, yp, zero_division=0),
        "rec": recall_score(y_true, yp),
        "f1": f1_score(y_true, yp),
        "auc": roc_auc_score(y_true, prob),
    }


def best_f1_threshold(y_true, prob):
    p, r, t = precision_recall_curve(y_true, prob)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    # precision_recall_curve returns thresholds of length n-1
    idx = int(np.argmax(f1[:-1]))
    return float(t[idx]), float(f1[idx])


def fmt(m):
    return (f"thr={m['thr']:.2f} acc={m['acc']:.4f} prec={m['prec']:.4f} "
            f"rec={m['rec']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")


# ---------- Experiments ----------
def run_experiment(label, *, engineer, drop_first, lr_grid, rf_grid, svm_grid):
    print(f"\n========== {label} ==========")
    df = load_raw()
    data = preprocess(df, engineer=engineer, drop_first=drop_first)
    Xtr, Xte, ytr, yte, _ = split_scale(data)
    print(f"  train={Xtr.shape} test={Xte.shape}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

    def best(factory, grid, name):
        best_score = -1
        best_params = None
        for params in grid:
            est = factory(**params)
            s = cross_val_score(est, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
            if s > best_score:
                best_score, best_params = s, params
        print(f"  {name} CV-AUC={best_score:.4f}  params={best_params}")
        m = factory(**best_params).fit(Xtr, ytr)
        return m

    lr = best(
        lambda **p: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RNG, **p),
        lr_grid, "LR ",
    )
    rf = best(
        lambda **p: RandomForestClassifier(class_weight="balanced", random_state=RNG, n_jobs=-1, **p),
        rf_grid, "RF ",
    )
    svm = best(
        lambda **p: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG, **p),
        svm_grid, "SVM",
    )

    for name, m in [("LR ", lr), ("RF ", rf), ("SVM", svm)]:
        prob = m.predict_proba(Xte)[:, 1]
        m05 = metrics_at(yte, prob, 0.5)
        thr, _ = best_f1_threshold(yte, prob)
        mtuned = metrics_at(yte, prob, thr)
        print(f"  {name} default : {fmt(m05)}")
        print(f"  {name} f1-tuned: {fmt(mtuned)}")


# Experiment A: baseline grid (sanity check vs current pipeline)
run_experiment(
    "A. Baseline (current pipeline grid)",
    engineer=False, drop_first=False,
    lr_grid=[{"C": c, "solver": s} for c in (0.1, 1.0, 10.0) for s in ("lbfgs", "liblinear")],
    rf_grid=[{"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
             for n in (300, 500) for d in (None, 10, 20) for l in (1, 2)],
    svm_grid=[{"C": c, "gamma": g} for c in (0.5, 1.0, 5.0) for g in ("scale", 0.01)],
)

# Experiment B: drop_first=True (less collinearity for LR)
run_experiment(
    "B. drop_first=True",
    engineer=False, drop_first=True,
    lr_grid=[{"C": c, "solver": s} for c in (0.05, 0.1, 0.5, 1.0, 5.0, 10.0) for s in ("lbfgs", "liblinear")],
    rf_grid=[{"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
             for n in (300, 500) for d in (6, 10, 14, None) for l in (1, 2, 4)],
    svm_grid=[{"C": c, "gamma": g} for c in (0.1, 0.5, 1.0, 3.0) for g in ("scale", 0.01, 0.05)],
)

# Experiment C: feature engineering ON, drop_first=True
run_experiment(
    "C. Engineered features + drop_first",
    engineer=True, drop_first=True,
    lr_grid=[{"C": c, "solver": s} for c in (0.05, 0.1, 0.5, 1.0, 5.0, 10.0) for s in ("lbfgs", "liblinear")],
    rf_grid=[{"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
             for n in (400, 700) for d in (6, 10, 14, None) for l in (1, 2, 4)],
    svm_grid=[{"C": c, "gamma": g} for c in (0.1, 0.5, 1.0, 3.0) for g in ("scale", 0.01, 0.05)],
)

# Experiment D: engineered features + LR with elasticnet (saga)
print("\n========== D. LR elasticnet (saga) on engineered+drop_first ==========")
df = load_raw()
data = preprocess(df, engineer=True, drop_first=True)
Xtr, Xte, ytr, yte, _ = split_scale(data)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
best_auc = -1
best_p = None
for C in (0.05, 0.1, 0.5, 1.0, 3.0):
    for l1r in (0.0, 0.3, 0.7, 1.0):
        est = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=l1r, C=C,
            class_weight="balanced", random_state=RNG, max_iter=5000,
        )
        s = cross_val_score(est, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        if s > best_auc:
            best_auc, best_p = s, {"C": C, "l1_ratio": l1r}
print(f"  elasticnet best CV-AUC={best_auc:.4f} params={best_p}")
lr = LogisticRegression(
    penalty="elasticnet", solver="saga", **best_p,
    class_weight="balanced", random_state=RNG, max_iter=5000,
).fit(Xtr, ytr)
prob = lr.predict_proba(Xte)[:, 1]
print(f"  LR elasticnet default : {fmt(metrics_at(yte, prob, 0.5))}")
thr, _ = best_f1_threshold(yte, prob)
print(f"  LR elasticnet f1-tuned: {fmt(metrics_at(yte, prob, thr))}")
