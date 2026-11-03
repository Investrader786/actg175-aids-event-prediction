"""Shared helpers for the ACTG 175 event-prediction project.

Both the training notebook and the Streamlit app import this module, so the
feature schema and the six evaluation metrics are defined in exactly one place.
That is what guarantees the numbers in README.md match the numbers the app
shows when test_data.csv is uploaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------
TARGET_COL = "cid"  # 1 = AIDS-defining event or death during follow-up
POSITIVE_CLASS = 1
CLASS_ORDER = [0, 1]
CLASS_LABELS = {0: "No event (censored)", 1: "AIDS event / death"}

# 21 input features used by every model, grouped by how they are preprocessed.
CONTINUOUS_FEATURES = ["age", "wtkg", "karnof", "preanti", "cd40", "cd420", "cd80", "cd820"]
BINARY_FEATURES = ["hemo", "homo", "drugs", "oprior", "z30", "race", "gender",
                   "str2", "symptom", "treat", "offtrt"]
NOMINAL_FEATURES = ["trt", "strat"]
FEATURE_COLUMNS = CONTINUOUS_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES

FEATURE_DESCRIPTIONS = {
    "age": "Age in years at baseline",
    "wtkg": "Weight in kg at baseline",
    "karnof": "Karnofsky performance score (0-100; higher = better daily functioning)",
    "preanti": "Days of antiretroviral therapy before the trial",
    "cd40": "CD4 count at baseline (cells/mm3)",
    "cd420": "CD4 count at 20 +/- 5 weeks",
    "cd80": "CD8 count at baseline (cells/mm3)",
    "cd820": "CD8 count at 20 +/- 5 weeks",
    "hemo": "Haemophilia (0 = no, 1 = yes)",
    "homo": "Homosexual activity (0 = no, 1 = yes)",
    "drugs": "History of intravenous drug use (0 = no, 1 = yes)",
    "oprior": "Non-ZDV antiretroviral therapy before the trial (0 = no, 1 = yes)",
    "z30": "ZDV in the 30 days before the trial (0 = no, 1 = yes)",
    "race": "Race (0 = white, 1 = non-white)",
    "gender": "Gender (0 = female, 1 = male)",
    "str2": "Antiretroviral history (0 = naive, 1 = experienced)",
    "symptom": "Symptomatic at baseline (0 = no, 1 = yes)",
    "treat": "Treatment indicator (0 = ZDV only, 1 = any other arm)",
    "offtrt": "Went off treatment before 96 +/- 5 weeks (0 = no, 1 = yes)",
    "trt": "Treatment arm (0 = ZDV only, 1 = ZDV + ddI, 2 = ZDV + zalcitabine, 3 = ddI only)",
    "strat": "Prior-therapy stratum (1 = naive, 2 = >1 to 52 weeks, 3 = >52 weeks)",
}
TARGET_DESCRIPTION = "1 = AIDS-defining event or death during follow-up, 0 = censored (no event observed)"

# Columns in the raw UCI table that are deliberately NOT used, with the reason.
DROPPED_COLUMNS = {
    "pidnum": "patient identifier - carries no clinical signal",
    "time": "days until event or censoring - it encodes the outcome itself (target leakage)",
    "zprior": "constant - every patient has the value 1",
}

# Display name (exactly as in the README tables) -> saved pipeline file in model/
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
METRIC_CONVENTION = (
    "Binary task, positive class = 1 (AIDS event / death). Precision, Recall and F1 "
    "use average='binary' with pos_label=1; AUC is ROC-AUC on the predicted probability "
    "of class 1; MCC is the Matthews correlation coefficient."
)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Predicted probability of the positive class (column chosen via classes_)."""
    proba = model.predict_proba(X)
    return proba[:, list(model.classes_).index(POSITIVE_CLASS)]


def score_predictions(y_true, y_pred, y_score) -> dict:
    """The six assignment metrics. AUC is NaN when y_true holds a single class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    auc = roc_auc_score(y_true, y_score) if np.unique(y_true).size == 2 else float("nan")
    common = dict(pos_label=POSITIVE_CLASS, average="binary", zero_division=0)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, **common),
        "Recall": recall_score(y_true, y_pred, **common),
        "F1": f1_score(y_true, y_pred, **common),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_model(model, X: pd.DataFrame, y) -> tuple[dict, np.ndarray, np.ndarray]:
    """Predict with an already-fitted pipeline and score it. Never fits anything."""
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    return score_predictions(y, y_pred, y_score), y_pred, y_score


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    names = [CLASS_LABELS[c] for c in CLASS_ORDER]
    return pd.DataFrame(cm, index=[f"Actual: {n}" for n in names],
                        columns=[f"Predicted: {n}" for n in names])


def report_frame(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, labels=CLASS_ORDER,
        target_names=[CLASS_LABELS[c] for c in CLASS_ORDER],
        output_dict=True, zero_division=0,
    )
    frame = pd.DataFrame(report).T
    if "accuracy" in frame.index:  # sklearn stores accuracy as a bare float; show it like its text report
        frame.loc["accuracy", ["precision", "recall"]] = np.nan
        frame.loc["accuracy", "support"] = frame.loc["macro avg", "support"]
    frame["support"] = frame["support"].astype(int)
    return frame


# --------------------------------------------------------------------------
# Input validation for uploaded test files
# --------------------------------------------------------------------------
def check_schema(df: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.Series | None, list[str], list[str]]:
    """Validate an uploaded table against the training schema.

    Returns (X, y, problems, notes). If `problems` is non-empty the file cannot be
    scored and X / y are None. `notes` are informational (e.g. ignored columns).
    """
    problems: list[str] = []
    notes: list[str] = []
    if df is None or df.empty:
        return None, None, ["The file has no rows."], notes

    df = df.rename(columns=lambda c: str(c).strip())
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        problems.append("Missing feature column(s): " + ", ".join(missing))
    if TARGET_COL not in df.columns:
        problems.append(f"Missing target column '{TARGET_COL}' - it is needed to compute the metrics.")
    if problems:
        return None, None, problems, notes

    extra = [c for c in df.columns if c not in FEATURE_COLUMNS + [TARGET_COL]]
    if extra:
        notes.append("Ignored column(s) not used by the models: " + ", ".join(extra))

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_text = [c for c in FEATURE_COLUMNS if X[c].isna().sum() > df[c].isna().sum()]
    if bad_text:
        problems.append("Non-numeric values found in: " + ", ".join(bad_text))
    if X.isna().any().any() and not bad_text:
        notes.append("Some feature values are blank; the saved pipelines impute them "
                     "with statistics learned from the training data.")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y.isna().any():
        problems.append(f"Target column '{TARGET_COL}' has {int(y.isna().sum())} blank or non-numeric value(s).")
    else:
        unseen = sorted(set(y.unique()) - set(CLASS_ORDER))
        if unseen:
            problems.append(f"Target column '{TARGET_COL}' contains label(s) the models never saw: {unseen}. "
                            f"Expected only {CLASS_ORDER}.")
    if problems:
        return None, None, problems, notes
    return X, y.astype(int), problems, notes
