"""
Risk Dashboard
==============
Full AI pipeline orchestrator for banking loan risk assessment.

Dataset:  Loan Prediction Dataset (Analytics Vidhya / Kaggle)
Source:   https://github.com/shrikant-temburwar/Loan-Prediction-Dataset
Citation: Analytics Vidhya. (2014). Loan Prediction Problem Dataset.
          Retrieved from https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

Executes:
  1. Real dataset loading and preprocessing (614 loan applications)
  2. Random Forest model training with GridSearchCV + 5-fold cross-validation
  3. Model evaluation: accuracy, precision, recall, F1, ROC-AUC, classification report
  4. Visualisation: feature importance, ROC curve, confusion matrix, risk distribution
  5. Per-application predictions with feature-level explainability
  6. Document verification (simulated OCR pipeline)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Ensure ai_solution directory is on path
sys.path.insert(0, os.path.dirname(__file__))

from ai_loan_assessor import (
    load_and_preprocess_data,
    train_model,
    evaluate_model,
    get_feature_importance,
    predict_application,
    engineer_features,
    get_feature_columns,
    DATASET_CITATION,
)
from document_verifier import verify_application_documents

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Visual styling
# ---------------------------------------------------------------------------

def _setup_style():
    plt.rcParams.update({
        "figure.facecolor":  "#FAFAFA",
        "axes.facecolor":    "#FAFAFA",
        "axes.edgecolor":    "#CCCCCC",
        "axes.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "font.family":       "DejaVu Sans",
        "grid.color":        "#E0E0E0",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
    })


PALETTE = {
    "blue":        "#2563EB",
    "green":       "#16A34A",
    "red":         "#DC2626",
    "orange":      "#D97706",
    "light_blue":  "#BFDBFE",
    "light_green": "#BBF7D0",
    "bg":          "#FAFAFA",
}


# ---------------------------------------------------------------------------
# Chart 1 – Feature Importance (Gini)
# ---------------------------------------------------------------------------

def plot_feature_importance(df_imp: pd.DataFrame, save_path: str):
    """Horizontal bar chart of Random Forest Gini feature importances."""
    fig, ax = plt.subplots(figsize=(10, 7))

    n    = len(df_imp)
    cmap = plt.cm.Blues
    colours = [cmap(0.35 + 0.65 * (1 - i / n)) for i in range(n)]

    bars = ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1],
                   color=colours[::-1], edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, df_imp["importance"][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#374151")

    ax.set_xlabel("Gini Importance", fontsize=10)
    ax.set_title(
        "Feature Importance — Random Forest Loan Risk Model\n"
        "(Analytics Vidhya Loan Prediction Dataset, n=614)",
        pad=12
    )
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="x", alpha=0.5)

    # Permutation importance overlay if available
    if "perm_importance" in df_imp.columns:
        ax2 = ax.twiny()
        ax2.set_xlabel("Permutation Importance (ROC-AUC drop)", fontsize=9,
                       color=PALETTE["orange"])
        perm_vals = df_imp["perm_importance"][::-1].values
        ax2.errorbar(
            perm_vals,
            range(n),
            xerr=df_imp["perm_std"][::-1].values if "perm_std" in df_imp.columns else None,
            fmt="D", color=PALETTE["orange"], markersize=5, linewidth=1.2,
            label="Permutation Imp.", capsize=3,
        )
        ax2.tick_params(axis="x", labelcolor=PALETTE["orange"], labelsize=8)
        ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart saved] {save_path}")


# ---------------------------------------------------------------------------
# Chart 2 – ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(metrics: dict, save_path: str):
    """ROC curve with AUC annotation and shaded area."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(metrics["fpr"], metrics["tpr"],
            color=PALETTE["blue"], lw=2.2,
            label=f"Loan Risk Model  (AUC = {metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.2,
            linestyle="--", label="Random Classifier (AUC = 0.5000)")

    ax.fill_between(metrics["fpr"], metrics["tpr"],
                    alpha=0.12, color=PALETTE["blue"])

    # Annotate best threshold region
    ax.annotate(
        f"AUC = {metrics['roc_auc']:.4f}",
        xy=(0.5, 0.5), xytext=(0.55, 0.35),
        fontsize=11, color=PALETTE["blue"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["blue"]),
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "ROC Curve — Loan Approval Prediction\n"
        "(Analytics Vidhya Loan Prediction Dataset)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.5)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart saved] {save_path}")


# ---------------------------------------------------------------------------
# Chart 3 – Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(metrics: dict, save_path: str):
    """Annotated confusion matrix heatmap with percentage labels."""
    cm    = metrics["confusion_matrix"]
    total = cm.sum()

    # Build annotated labels with counts + percentages
    annot = np.array(
        [[f"{v}\n({v/total:.1%})" for v in row] for row in cm]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=["Denied (0)", "Approved (1)"],
        yticklabels=["Denied (0)", "Approved (1)"],
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label",  fontsize=10)
    ax.set_ylabel("Actual Label",     fontsize=10)
    ax.set_title(
        "Confusion Matrix — Test Set Predictions\n"
        f"Accuracy: {metrics['accuracy']:.4f}  |  F1: {metrics['f1']:.4f}"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart saved] {save_path}")


# ---------------------------------------------------------------------------
# Chart 4 – Risk Score Distribution
# ---------------------------------------------------------------------------

def plot_risk_distribution(metrics: dict, save_path: str):
    """Histogram + KDE of predicted approval probabilities by true label."""
    from scipy.stats import gaussian_kde

    y_proba = metrics["y_proba"]
    y_test  = metrics["y_test"]

    approved_proba = y_proba[y_test == 1]
    denied_proba   = y_proba[y_test == 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(denied_proba,   bins=30, alpha=0.50, color=PALETTE["red"],
            label=f"Actual: Denied  (n={len(denied_proba)})",
            density=True, edgecolor="white")
    ax.hist(approved_proba, bins=30, alpha=0.50, color=PALETTE["green"],
            label=f"Actual: Approved (n={len(approved_proba)})",
            density=True, edgecolor="white")

    # KDE overlays
    for proba, colour in [(denied_proba, PALETTE["red"]),
                          (approved_proba, PALETTE["green"])]:
        if len(proba) > 2:
            kde = gaussian_kde(proba, bw_method=0.18)
            xs  = np.linspace(0, 1, 300)
            ax.plot(xs, kde(xs), color=colour, lw=2.2)

    ax.axvline(0.50, color="#374151", lw=1.8, linestyle="--",
               label="Decision threshold (0.50)")

    # Risk band shading
    ax.axvspan(0, 0.40,  alpha=0.04, color=PALETTE["red"])
    ax.axvspan(0.60, 1,  alpha=0.04, color=PALETTE["green"])

    ax.set_xlabel("Predicted Approval Probability")
    ax.set_ylabel("Density")
    ax.set_title(
        "Risk Score Distribution — Test Set\n"
        "(Analytics Vidhya Loan Prediction Dataset)"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    ax.set_xlim([0, 1])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart saved] {save_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_pipeline():
    _setup_style()
    sep   = "=" * 72
    dash  = "-" * 72

    print(sep)
    print("   BANKING AI LOAN RISK ASSESSMENT — REAL DATASET PIPELINE")
    print(sep)
    print(f"\n  Dataset : {DATASET_CITATION}")
    print(f"  Source  : https://github.com/shrikant-temburwar/Loan-Prediction-Dataset")
    print()

    # ── Step 1 ─ Load and preprocess real dataset ──────────────────────────
    print(f"[Step 1/6]  Loading and preprocessing real loan dataset...")
    df = load_and_preprocess_data()
    print(f"\n  Dataset overview:")
    print(f"  {'Total applications':30s}: {len(df)}")
    print(f"  {'Approved (Y)':30s}: {df['approved'].sum()}  ({df['approved'].mean():.1%})")
    print(f"  {'Denied (N)':30s}: {(1-df['approved']).sum()}  ({(1-df['approved']).mean():.1%})")
    print(f"  {'Features (raw + engineered)':30s}: {len(get_feature_columns())}")
    print(f"\n  Column summary:")
    print(f"  {'Column':<25} {'Type':<10} {'Non-Null':>8}  {'Unique':>8}")
    print("  " + "-" * 55)
    for col in df.columns:
        print(f"  {col:<25} {str(df[col].dtype):<10} "
              f"{df[col].notna().sum():>8}  {df[col].nunique():>8}")

    # ── Step 2 ─ Train model ───────────────────────────────────────────────
    print(f"\n[Step 2/6]  Training Random Forest with GridSearchCV + 5-fold CV...")
    pipeline, X_test, y_test, feat_cols, cv_scores = train_model(df)

    print(f"\n  Cross-validation results (5-fold accuracy on training set):")
    for i, score in enumerate(cv_scores, 1):
        print(f"    Fold {i}: {score:.4f}")
    print(f"  Mean : {cv_scores.mean():.4f}  |  Std : {cv_scores.std():.4f}")

    # ── Step 3 ─ Evaluate ─────────────────────────────────────────────────
    print(f"\n[Step 3/6]  Evaluating model on held-out test set ({len(y_test)} samples)...")
    metrics = evaluate_model(pipeline, X_test, y_test)

    print(f"\n  {'Metric':<20} {'Value':>10}")
    print("  " + "-" * 32)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        label = k.upper() if k == "roc_auc" else k.capitalize()
        print(f"  {label:<20} {metrics[k]:>10.4f}")

    print(f"\n  Full Classification Report:")
    print("  " + dash)
    for line in metrics["classification_report"].split("\n"):
        print(f"  {line}")

    # ── Step 4 ─ Feature importance ────────────────────────────────────────
    print(f"\n[Step 4/6]  Computing feature importances (Gini + permutation)...")
    df_imp = get_feature_importance(pipeline, feat_cols, X_test, y_test)

    print(f"\n  {'Feature':<25} {'Gini Imp.':>10}  {'Perm Imp.':>10}  {'Perm Std':>9}")
    print("  " + "-" * 58)
    for _, row in df_imp.iterrows():
        perm = f"{row.get('perm_importance', float('nan')):.4f}"
        pstd = f"{row.get('perm_std', float('nan')):.4f}"
        print(f"  {row['feature']:<25} {row['importance']:>10.4f}  {perm:>10}  {pstd:>9}")

    # ── Step 5 ─ Visualisations ────────────────────────────────────────────
    print(f"\n[Step 5/6]  Generating visualisation charts...")
    plot_feature_importance(df_imp,
        os.path.join(OUT_DIR, "feature_importance.png"))
    plot_roc_curve(metrics,
        os.path.join(OUT_DIR, "roc_curve.png"))
    plot_confusion_matrix(metrics,
        os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_risk_distribution(metrics,
        os.path.join(OUT_DIR, "risk_distribution.png"))

    # ── Step 6 ─ Sample application predictions ────────────────────────────
    print(f"\n[Step 6/6]  Predicting sample applications...")
    print(f"\n{sep}")
    print("  SAMPLE APPLICATION PREDICTIONS")
    print(sep)

    apps_path = os.path.join(OUT_DIR, "sample_new_applications.json")
    with open(apps_path) as f:
        applications = json.load(f)

    # Field decoding helpers
    gender_map   = {0: "Female", 1: "Male"}
    married_map  = {0: "Single", 1: "Married"}
    edu_map      = {0: "Not Graduate", 1: "Graduate"}
    emp_map      = {0: "Salaried", 1: "Self-Employed"}
    area_map     = {0: "Rural", 1: "Semiurban", 2: "Urban"}
    credit_map   = {0: "No (0)", 1: "Yes (1)"}

    for app in applications:
        print(f"\n  Application ID   : {app['application_id']}")
        print(f"  Applicant        : {app['applicant_name']}")
        print(f"  Gender           : {gender_map.get(app.get('gender', 1), 'N/A')}")
        print(f"  Married          : {married_map.get(app.get('married', 1), 'N/A')}")
        print(f"  Education        : {edu_map.get(app.get('education', 1), 'N/A')}")
        print(f"  Self-Employed    : {emp_map.get(app.get('self_employed', 0), 'N/A')}")
        print(f"  Applicant Income : {app.get('applicant_income', 0):,.0f}/mo")
        print(f"  Co-App Income    : {app.get('coapplicant_income', 0):,.0f}/mo")
        print(f"  Loan Amount      : {app.get('loan_amount', 0):,.0f}k")
        print(f"  Loan Term        : {app.get('loan_amount_term', 360):.0f} months")
        print(f"  Credit History   : {credit_map.get(app.get('credit_history', 1), 'N/A')}")
        print(f"  Property Area    : {area_map.get(app.get('property_area', 1), 'N/A')}")
        print(f"  Dependents       : {app.get('dependents', 0)}")
        print(f"  Notes            : {app.get('loan_notes', 'N/A')}")

        result = predict_application(pipeline, app, feat_cols)
        print(f"\n  >>> AI DECISION        : {result['decision']}")
        print(f"  >>> Approval Prob      : {result['approval_probability']:.4f}")
        print(f"  >>> Risk Level         : {result['risk_level']}")
        print("  >>> Top Feature Contributions:")
        for line in result["explanation"]:
            print(f"       • {line}")

        # Document verification
        doc_result = verify_application_documents(
            app,
            doc_types=["pay_stub", "credit_report"],
            seed=hash(app["application_id"]) % 10_000,
        )
        status  = doc_result["overall_status"]
        conf    = doc_result["overall_confidence"]
        flagged = doc_result["flagged_for_human_review"]

        print(f"\n  >>> DOC VERIFICATION   : {status}  (confidence {conf:.2%})")
        if doc_result["issue_summary"]:
            for issue in doc_result["issue_summary"]:
                print(f"       [ISSUE] {issue}")
        if flagged:
            print("       *** FLAGGED FOR HUMAN REVIEW ***")
        print("  " + dash)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  PIPELINE COMPLETE — SUMMARY")
    print(sep)
    print(f"  Dataset          : Analytics Vidhya Loan Prediction (n=614)")
    print(f"  Source URL       : https://github.com/shrikant-temburwar/Loan-Prediction-Dataset")
    print(f"  Model            : Random Forest (GridSearchCV tuned)")
    print(f"  CV Accuracy      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC Score    : {metrics['roc_auc']:.4f}")
    print(f"  F1 Score         : {metrics['f1']:.4f}")
    print(f"  Precision        : {metrics['precision']:.4f}")
    print(f"  Recall           : {metrics['recall']:.4f}")
    print(f"  Charts saved to  : {OUT_DIR}/")
    print(f"  Applications     : {len(applications)} scored")
    print(sep)


if __name__ == "__main__":
    run_pipeline()
