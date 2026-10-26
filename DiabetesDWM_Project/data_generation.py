"""
data_generation.py
==================
Generates a realistic synthetic diabetes dataset with ~1000 samples.

Features are modeled after common diabetes risk-factor distributions,
with controlled noise and realistic class imbalance (~35% diabetic).

Usage:
    python data_generation.py
    → Produces `diabetes_dataset.csv` in the current directory.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 1200  # total samples
DIABETES_RATIO = 0.35  # ~35 % positive → slight class imbalance

OUTPUT_PATH = Path(__file__).parent / "diabetes_dataset.csv"


def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip values and round to 1 decimal."""
    return np.round(np.clip(arr, lo, hi), 1)


def generate_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a synthetic diabetes dataset.

    Positive‑class samples are drawn from distributions shifted towards
    higher glucose, BMI, age, etc. to mirror real medical correlations.
    """
    rng = np.random.default_rng(seed)

    n_pos = int(n * DIABETES_RATIO)
    n_neg = n - n_pos

    # --- Negative class (no diabetes) ------------------------------------
    neg = {
        # 0 = Male, 1 = Female
        "Gender": rng.binomial(1, 0.50, size=n_neg),
        "Glucose": _clip(rng.normal(100, 18, n_neg), 44, 199),
        "BloodPressure": _clip(rng.normal(70, 10, n_neg), 30, 122),
        "SkinThickness": _clip(rng.normal(25, 8, n_neg), 7, 60),
        "Insulin": _clip(rng.normal(100, 50, n_neg), 14, 846),
        "BMI": _clip(rng.normal(28, 5, n_neg), 18, 55),
        "DiabetesPedigreeFunction": _clip(rng.exponential(0.35, n_neg), 0.08, 2.42),
        "Age": rng.integers(21, 60, size=n_neg),
    }

    # --- Positive class (diabetes) ----------------------------------------
    pos = {
        # Slightly higher female proportion in positive class for variety
        "Gender": rng.binomial(1, 0.56, size=n_pos),
        "Glucose": _clip(rng.normal(140, 25, n_pos), 56, 199),
        "BloodPressure": _clip(rng.normal(75, 12, n_pos), 30, 122),
        "SkinThickness": _clip(rng.normal(30, 10, n_pos), 7, 63),
        "Insulin": _clip(rng.normal(180, 80, n_pos), 14, 846),
        "BMI": _clip(rng.normal(34, 6, n_pos), 18, 60),
        "DiabetesPedigreeFunction": _clip(rng.exponential(0.55, n_pos), 0.08, 2.42),
        "Age": rng.integers(25, 75, size=n_pos),
    }

    df_neg = pd.DataFrame(neg)
    df_neg["Outcome"] = 0

    df_pos = pd.DataFrame(pos)
    df_pos["Outcome"] = 1

    df = pd.concat([df_neg, df_pos], ignore_index=True)

    # --- Inject realistic missingness (~5 %) on select columns ------------
    cols_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_missing:
        mask = rng.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    # Shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_dataset()

    # Quick sanity checks
    print("=" * 60)
    print("  Synthetic Diabetes Dataset Generation")
    print("=" * 60)
    print(f"  Samples        : {len(df)}")
    print(f"  Features       : {list(df.columns[:-1])}")
    print(f"  Class split    : {df['Outcome'].value_counts().to_dict()}")
    print(f"  Missing values :\n{df.isnull().sum().to_string()}")
    print("=" * 60)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  ✅  Dataset saved → {OUTPUT_PATH}")
