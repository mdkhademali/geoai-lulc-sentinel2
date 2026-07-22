"""
model.py
========
Model training, hyperparameter tuning, and persistence for LULC
classification: Random Forest, XGBoost, LightGBM, SVM.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      StratifiedKFold, cross_val_score,
                                      learning_curve, validation_curve)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


def get_base_models(random_state: int = 42) -> Dict[str, object]:
    """Return the four candidate model families with sane default hyperparameters."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_state
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, eval_metric="mlogloss", n_jobs=-1,
            random_state=random_state,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=random_state, verbose=-1,
        ),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=random_state),
    }


def train_and_compare(models: Dict[str, object], X_train, y_train, X_test, y_test,
                       cv_folds: int = 5) -> Tuple[Dict, "pd.DataFrame"]:
    """Fit each model, time it, score cross-validated accuracy on train and
    held-out accuracy on test. Returns fitted models + a comparison table."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    fitted, rows = {}, []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="macro")

        fitted[name] = model
        rows.append({
            "Model": name,
            "CV Accuracy (mean)": cv_scores.mean(),
            "CV Accuracy (std)": cv_scores.std(),
            "Test Accuracy": test_acc,
            "Test Macro-F1": test_f1,
            "Train Time (s)": train_time,
        })
    comparison = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    return fitted, comparison


def tune_random_forest(X_train, y_train, search: str = "random", n_iter: int = 20,
                        random_state: int = 42):
    """Hyperparameter search for Random Forest via GridSearchCV or RandomizedSearchCV."""
    param_grid = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    if search == "grid":
        small_grid = {k: v[:2] for k, v in param_grid.items()}
        searcher = GridSearchCV(base, small_grid, cv=skf, scoring="accuracy", n_jobs=-1)
    else:
        searcher = RandomizedSearchCV(base, param_grid, n_iter=n_iter, cv=skf,
                                       scoring="accuracy", n_jobs=-1, random_state=random_state)
    searcher.fit(X_train, y_train)
    return searcher


def get_learning_curve(model, X, y, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy",
    )
    return train_sizes, train_scores, val_scores


def get_validation_curve(model, X, y, param_name, param_range, cv=5):
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, n_jobs=-1, scoring="accuracy",
    )
    return train_scores, val_scores


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
