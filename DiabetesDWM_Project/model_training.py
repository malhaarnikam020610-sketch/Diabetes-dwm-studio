"""
model_training.py
=================
Loads the synthetic diabetes dataset, performs EDA, trains an XGBoost
classifier with hyperparameter tuning and class‑imbalance handling,
evaluates the model, and saves the trained pipeline.

Usage:
    python model_training.py
    → Produces evaluation plots and `diabetes_model.pkl`.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non‑interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "diabetes_dataset.csv"
EXTRA_DATA_CANDIDATES = [
    BASE_DIR / "diabetes.csv",
    Path.home() / "Downloads" / "diabetes.csv",
]
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"primary": "#6C63FF", "secondary": "#FF6584", "bg": "#1E1E2E"}


# ===========================================================================
# 1.  LOAD & PREPROCESS
# ===========================================================================
def _load_single_dataset(path: Path) -> pd.DataFrame:
    """Load one dataset and map columns to the model schema."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Compatibility for common public diabetes schema.
    if "Pregnancies" in df.columns and "Gender" not in df.columns:
        # The public Pima dataset contains only female participants.
        df["Gender"] = 0

    required_cols = [
        "Gender",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing required columns: {missing}"
        )
    return df[required_cols].copy()


def load_and_preprocess(path: Path) -> pd.DataFrame:
    """Load one or more CSV files and impute missing values with medians."""
    dataframes = []
    sources = []

    dataframes.append(_load_single_dataset(path))
    sources.append(path)

    for extra_path in EXTRA_DATA_CANDIDATES:
        if extra_path.exists() and extra_path.resolve() != path.resolve():
            dataframes.append(_load_single_dataset(extra_path))
            sources.append(extra_path)

    df = pd.concat(dataframes, ignore_index=True)

    print("\n📁  Data sources used:")
    for src in sources:
        print(f"  - {src}")

    # In many clinical datasets, zero indicates missing for these fields.
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

    print("\n📊  Raw dataset shape:", df.shape)
    print("\n  Missing values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

    # Median imputation — robust to outliers
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    print("\n  ✅  Missing values after imputation:", df.isnull().sum().sum())
    return df


# ===========================================================================
# 2.  EXPLORATORY DATA ANALYSIS
# ===========================================================================
def run_eda(df: pd.DataFrame) -> None:
    """Generate EDA outputs: summary stats, correlation heatmap, distributions."""
    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Summary statistics
    print("\n  Summary Statistics:")
    print(df.describe().round(2).to_string())

    # Class distribution
    print("\n  Class Distribution:")
    counts = df["Outcome"].value_counts()
    for label, count in counts.items():
        tag = "No Diabetes" if label == 0 else "Diabetes"
        print(f"    {tag} ({label}): {count}  ({count/len(df)*100:.1f}%)")

    # --- Correlation heatmap -----------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
        linewidths=0.5, square=True, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\n  📈  Saved → {PLOTS_DIR / 'correlation_heatmap.png'}")

    # --- Feature distributions by class ------------------------------------
    features = [c for c in df.columns if c != "Outcome"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for idx, (feat, ax) in enumerate(zip(features, axes.ravel())):
        for label, color in [(0, COLORS["primary"]), (1, COLORS["secondary"])]:
            subset = df[df["Outcome"] == label][feat]
            ax.hist(subset, bins=30, alpha=0.55, color=color,
                    label="No Diabetes" if label == 0 else "Diabetes")
        ax.set_title(feat, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle("Feature Distributions by Outcome", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈  Saved → {PLOTS_DIR / 'feature_distributions.png'}")


# ===========================================================================
# 3.  MODEL TRAINING
# ===========================================================================
def train_model(df: pd.DataFrame):
    """
    Train an XGBoost classifier with grid‑search hyperparameter tuning,
    class‑imbalance weighting, and feature scaling.

    Returns: (pipeline, X_test, y_test, best_params)
    """
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # 80 / 20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"\n  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # Compute scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    spw = neg_count / pos_count
    print(f"  scale_pos_weight: {spw:.2f}")

    # Pipeline: StandardScaler → XGBClassifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Hyperparameter grid
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    print("\n  🔍  Running GridSearchCV (3‑fold) — this may take a moment …")
    grid = GridSearchCV(
        pipe, param_grid,
        cv=3, scoring="f1", n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)

    print(f"  ✅  Best F1 (CV): {grid.best_score_:.4f}")
    print(f"  ✅  Best params : {grid.best_params_}")

    return grid.best_estimator_, X_test, y_test, grid.best_params_


# ===========================================================================
# 4.  EVALUATION
# ===========================================================================
def evaluate_model(pipeline, X_test, y_test) -> None:
    """Print metrics, confusion matrix, and ROC curve."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1‑Score  : {f1_score(y_test, y_pred):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    # --- Confusion Matrix --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  📈  Saved → {PLOTS_DIR / 'confusion_matrix.png'}")

    # --- ROC Curve ---------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLORS["primary"], lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"  📈  Saved → {PLOTS_DIR / 'roc_curve.png'}")
    print(f"\n  🏆  AUC Score: {roc_auc:.4f}")


# ===========================================================================
# 5.  FEATURE IMPORTANCE
# ===========================================================================
def plot_feature_importance(pipeline, feature_names: list) -> None:
    """Extract and plot XGBoost feature importances."""
    clf = pipeline.named_steps["clf"]
    importances = clf.feature_importances_

    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))
    ax.barh(range(len(idx)), importances[idx], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("XGBoost Feature Importance", fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  📈  Saved → {PLOTS_DIR / 'feature_importance.png'}")


# ===========================================================================
# 6.  SAVE MODEL
# ===========================================================================
def save_model(pipeline, path: Path) -> None:
    """Persist the trained pipeline to disk."""
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  💾  Model saved → {path}  ({path.stat().st_size / 1024:.1f} KB)")


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    print("=" * 60)
    print("  DIABETES PREDICTION — XGBoost Pipeline")
    print("=" * 60)

    # Step 1: Load & clean
    df = load_and_preprocess(DATA_PATH)

    # Step 2: EDA
    run_eda(df)

    # Step 3: Train
    pipeline, X_test, y_test, best_params = train_model(df)

    # Step 4: Evaluate
    evaluate_model(pipeline, X_test, y_test)

    # Step 5: Feature importance
    feature_names = [c for c in df.columns if c != "Outcome"]
    plot_feature_importance(pipeline, feature_names)

    # Step 6: Save
    save_model(pipeline, MODEL_PATH)

    print("\n" + "=" * 60)
    print("  ✅  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
