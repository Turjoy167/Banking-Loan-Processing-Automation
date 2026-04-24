"""
AI Loan Assessor Module
=======================
ML-based credit risk scoring for banking loan applications.

Dataset:  Loan Prediction Dataset (Analytics Vidhya / Kaggle)
Source:   https://github.com/shrikant-temburwar/Loan-Prediction-Dataset
Citation: Analytics Vidhya. (2014). Loan Prediction Problem Dataset.
          Retrieved from https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

Uses a Random Forest classifier (with GridSearchCV hyperparameter tuning)
trained on the real Loan Prediction Dataset to:
  - Predict loan approval / default risk
  - Compute per-application probability scores with 5-fold cross-validation
  - Provide feature-importance and permutation-importance explainability
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset path (relative to this file)
# ---------------------------------------------------------------------------
_DATA_PATH = os.path.join(os.path.dirname(__file__), "loan_data.csv")

DATASET_CITATION = (
    "Analytics Vidhya Loan Prediction Dataset (2014). "
    "614 real loan applications with 12 features. "
    "Source: https://github.com/shrikant-temburwar/Loan-Prediction-Dataset"
)


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_data(filepath: str = _DATA_PATH) -> pd.DataFrame:
    """
    Load the real Loan Prediction Dataset and apply full preprocessing.

    Steps:
      1. Load CSV
      2. Drop Loan_ID identifier
      3. Impute missing values (mode for categoricals, median for numerics)
      4. Feature engineering (derived ratios, categorical encoding)
      5. Encode binary target (Loan_Status Y→1, N→0)

    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataset ready for model training.
    """
    df = pd.read_csv(filepath)
    print(f"[Dataset] Loaded '{filepath}'")
    print(f"          Shape: {df.shape}  |  Columns: {df.columns.tolist()}")

    # Drop identifier column
    df.drop(columns=["Loan_ID"], inplace=True, errors="ignore")

    # ── Missing value imputation ──────────────────────────────────────────
    # Categorical → mode imputation (use assignment-style for pandas CoW compat)
    cat_cols = ["Gender", "Married", "Dependents", "Self_Employed",
                "Loan_Amount_Term", "Credit_History"]
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # Numeric → median imputation
    num_cols = ["LoanAmount"]
    for col in num_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # ── Normalise / encode Dependents ────────────────────────────────────
    # Dependents is stored as '0','1','2','3+' — map '3+' → 3
    df["Dependents"] = df["Dependents"].astype(str).str.replace("+", "", regex=False)
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce").fillna(0).astype(int)

    # ── Encode binary categoricals ────────────────────────────────────────
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                  "Graduate": 1, "Not Graduate": 0, "Y": 1, "N": 0}
    for col in ["Gender", "Married", "Education", "Self_Employed", "Loan_Status"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # ── Property_Area ordinal encoding ───────────────────────────────────
    area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    df["Property_Area"] = df["Property_Area"].map(area_map).fillna(1)

    # ── Ensure Credit_History is integer ─────────────────────────────────
    df["Credit_History"] = df["Credit_History"].astype(float)

    # ── Rename for clarity ────────────────────────────────────────────────
    df.rename(columns={
        "ApplicantIncome":    "applicant_income",
        "CoapplicantIncome":  "coapplicant_income",
        "LoanAmount":         "loan_amount",
        "Loan_Amount_Term":   "loan_amount_term",
        "Credit_History":     "credit_history",
        "Property_Area":      "property_area",
        "Loan_Status":        "approved",
        "Gender":             "gender",
        "Married":            "married",
        "Dependents":         "dependents",
        "Education":          "education",
        "Self_Employed":      "self_employed",
    }, inplace=True)

    # ── Final numeric fillna for any residual nulls after encoding ──────────
    # (e.g. gender/married/self_employed rows that were originally NaN strings)
    for col in df.select_dtypes(include=[float, int]).columns:
        if col != "approved" and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"[Dataset] After preprocessing: {df.shape}")
    print(f"[Dataset] Approval rate: {df['approved'].mean():.1%}  "
          f"({df['approved'].sum()} approved / {(1-df['approved']).sum()} denied)")
    print(f"[Dataset] Missing values remaining: {df.isnull().sum().sum()}")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the preprocessed DataFrame.

    New features:
      - total_income        : applicant + co-applicant income
      - loan_to_income      : loan_amount / total_income (credit utilisation proxy)
      - income_per_person   : total_income / (1 + dependents)
      - emi_estimate        : simple EMI = loan_amount / loan_amount_term
      - emi_to_income_ratio : emi / monthly total income (affordability ratio)
      - log_loan_amount     : log-transform to reduce skew
      - log_total_income    : log-transform to reduce skew

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed loan data (output of load_and_preprocess_data or
        a single-row dict converted to DataFrame).

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered feature columns.
    """
    df = df.copy()

    df["total_income"]        = df["applicant_income"] + df["coapplicant_income"]
    df["loan_to_income"]      = df["loan_amount"] / (df["total_income"] + 1)
    df["income_per_person"]   = df["total_income"] / (1 + df["dependents"])
    df["emi_estimate"]        = df["loan_amount"] / df["loan_amount_term"].replace(0, 360)
    df["emi_to_income_ratio"] = df["emi_estimate"] / (df["total_income"] / 12 + 1)
    df["log_loan_amount"]     = np.log1p(df["loan_amount"])
    df["log_total_income"]    = np.log1p(df["total_income"])

    return df


def get_feature_columns() -> list:
    """Return the ordered list of numeric feature columns used for training."""
    return [
        # Raw features (encoded)
        "gender", "married", "dependents", "education", "self_employed",
        "applicant_income", "coapplicant_income", "loan_amount",
        "loan_amount_term", "credit_history", "property_area",
        # Engineered features
        "total_income", "loan_to_income", "income_per_person",
        "emi_estimate", "emi_to_income_ratio",
        "log_loan_amount", "log_total_income",
    ]


# ---------------------------------------------------------------------------
# Model training with GridSearchCV
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier with GridSearchCV hyperparameter tuning.

    Steps:
      1. Feature engineering
      2. Stratified 80/20 train-test split
      3. GridSearchCV (5-fold CV) over RF hyperparameters
      4. 5-fold cross-validation on full training set with best params

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset (output of load_and_preprocess_data).

    Returns
    -------
    tuple
        (pipeline, X_test, y_test, feat_cols, cv_scores)
          pipeline   – best fitted sklearn Pipeline (scaler + RF)
          X_test     – held-out feature array (numpy)
          y_test     – held-out labels (numpy)
          feat_cols  – list of feature column names
          cv_scores  – array of 5-fold CV accuracy scores
    """
    df_eng    = engineer_features(df)
    feat_cols = get_feature_columns()

    X = df_eng[feat_cols].values.astype(float)
    y = df_eng["approved"].values.astype(int)

    # Stratified train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[Split] Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"[Split] Train approval rate: {y_train.mean():.1%}  "
          f"| Test approval rate: {y_test.mean():.1%}")

    # Build pipeline
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Hyperparameter grid
    param_grid = {
        "clf__n_estimators":    [100, 200, 300],
        "clf__max_depth":       [6, 10, 15, None],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features":    ["sqrt", "log2"],
    }

    print("[GridSearchCV] Searching hyperparameter space (5-fold CV)...")
    gs = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    gs.fit(X_train, y_train)

    best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    print(f"[GridSearchCV] Best params : {best_params}")
    print(f"[GridSearchCV] Best CV AUC : {gs.best_score_:.4f}")

    pipeline = gs.best_estimator_

    # 5-fold cross-validation accuracy on full training set
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    print(f"[CV] 5-Fold Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  "
          f"| Scores: {[round(s, 4) for s in cv_scores]}")

    return pipeline, X_test, y_test, feat_cols, cv_scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute standard binary classification metrics.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc, confusion_matrix,
              fpr, tpr, y_proba, y_pred, y_test, classification_report
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "accuracy":              accuracy_score(y_test, y_pred),
        "precision":             precision_score(y_test, y_pred, zero_division=0),
        "recall":                recall_score(y_test, y_pred, zero_division=0),
        "f1":                    f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":               roc_auc_score(y_test, y_proba),
        "confusion_matrix":      confusion_matrix(y_test, y_pred),
        "fpr":                   fpr,
        "tpr":                   tpr,
        "y_proba":               y_proba,
        "y_pred":                y_pred,
        "y_test":                y_test,
        "classification_report": classification_report(
            y_test, y_pred,
            target_names=["Denied (0)", "Approved (1)"],
            zero_division=0,
        ),
    }

    print("\n[Metrics]")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k.capitalize():12s}: {metrics[k]:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Feature importance (Gini + permutation)
# ---------------------------------------------------------------------------

def get_feature_importance(pipeline, feature_names: list,
                            X_test: np.ndarray = None,
                            y_test: np.ndarray = None) -> pd.DataFrame:
    """
    Extract Gini-based feature importances from the Random Forest.
    If X_test and y_test are provided, also computes permutation importance.

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance, perm_importance (if X_test provided)
        Sorted by Gini importance descending.
    """
    rf  = pipeline.named_steps["clf"]
    imp = rf.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": imp})

    if X_test is not None and y_test is not None:
        perm_result = permutation_importance(
            pipeline, X_test, y_test,
            n_repeats=20, random_state=42, n_jobs=-1,
            scoring="roc_auc",
        )
        df_imp["perm_importance"] = perm_result.importances_mean
        df_imp["perm_std"]        = perm_result.importances_std

    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp


# ---------------------------------------------------------------------------
# Single-application prediction with explainability
# ---------------------------------------------------------------------------

def predict_application(pipeline, application: dict, feature_names: list) -> dict:
    """
    Score a single loan application and generate a human-readable explanation.

    The application dict must use the REAL dataset field names (post-preprocessing):
      gender, married, dependents, education, self_employed,
      applicant_income, coapplicant_income, loan_amount,
      loan_amount_term, credit_history, property_area

    Parameters
    ----------
    pipeline      : fitted sklearn Pipeline
    application   : dict with raw loan application fields
    feature_names : list of feature columns used during training

    Returns
    -------
    dict
        decision, approval_probability, risk_level, explanation
    """
    df_app = pd.DataFrame([application])
    df_app = engineer_features(df_app)

    # Ensure all feature columns are present
    for col in feature_names:
        if col not in df_app.columns:
            df_app[col] = 0

    X = df_app[feature_names].values.astype(float)

    proba    = pipeline.predict_proba(X)[0][1]   # P(approved=1)
    decision = "APPROVED" if proba >= 0.50 else "DENIED"

    if proba >= 0.80:
        risk_level = "LOW RISK"
    elif proba >= 0.60:
        risk_level = "MODERATE RISK"
    elif proba >= 0.40:
        risk_level = "HIGH RISK"
    else:
        risk_level = "VERY HIGH RISK"

    # Permutation-style contribution: importance × |scaled feature value|
    rf       = pipeline.named_steps["clf"]
    scaler   = pipeline.named_steps["scaler"]
    imp      = rf.feature_importances_
    X_scaled = scaler.transform(X)[0]

    contrib  = imp * np.abs(X_scaled)
    total    = contrib.sum()
    if total > 0:
        contrib = contrib / total

    top_idx = np.argsort(contrib)[::-1][:5]
    explanation = []
    for i in top_idx:
        raw_val   = X[0][i]
        direction = "positively" if X_scaled[i] > 0 else "negatively"
        explanation.append(
            f"'{feature_names[i]}' = {raw_val:.2f}  "
            f"({contrib[i]:.1%} weight, influenced decision {direction})"
        )

    return {
        "decision":             decision,
        "approval_probability": round(float(proba), 4),
        "risk_level":           risk_level,
        "explanation":          explanation,
    }
