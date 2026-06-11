"""
preprocessing.py

Cleaning, label encoding, and train/test splitting utilities for the
CICIDS2017 dataset.

Pipeline summary
-----------------
1. Replace inf/-inf with NaN, drop rows with NaN in feature columns.
2. Drop exact duplicate rows.
3. Drop columns that are constant (zero variance) -- these add no
   information and can break some scalers.
4. Build two label targets:
     - `label_binary`   : BENIGN vs ATTACK
     - `label_multiclass`: original attack category, with extremely
                            rare classes (<MIN_CLASS_COUNT samples)
                            grouped into "Other" so stratified
                            train/test splitting is possible.
5. Stratified train/test split on whichever target is requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns that are identifiers / leakage / not real traffic features
DROP_COLS = ["source_file", "Flow ID", "Source IP", "Destination IP",
              "Timestamp", "Fwd Header Length.1"]

MIN_CLASS_COUNT = 200  # classes with fewer rows than this get grouped into "Other"


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    label_classes: list[str]
    scaler: StandardScaler


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove inf/NaN rows, duplicates, and useless columns."""
    df = df.copy()

    # Drop columns we don't want as features (if present)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Replace inf/-inf with NaN, then drop rows containing NaN in
    # numeric feature columns. This handles the well-known
    # 'Flow Bytes/s' / 'Flow Packets/s' infinity issue.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_before = len(df)
    df = df.dropna(subset=numeric_cols)
    n_after = len(df)
    print(f"Dropped {n_before - n_after:,} rows containing inf/NaN "
          f"({(n_before - n_after) / n_before:.3%} of data)")

    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    print(f"Dropped {n_before - n_after:,} duplicate rows "
          f"({(n_before - n_after) / n_before:.3%} of data)")

    # Drop zero-variance columns
    zero_var_cols = [c for c in numeric_cols
                      if c in df.columns and df[c].nunique() <= 1]
    if zero_var_cols:
        print(f"Dropping {len(zero_var_cols)} zero-variance columns: "
              f"{zero_var_cols}")
        df = df.drop(columns=zero_var_cols)

    return df.reset_index(drop=True)


def add_label_columns(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """Add `label_binary` and `label_multiclass` columns."""
    df = df.copy()

    df["label_binary"] = np.where(df[label_col].str.upper() == "BENIGN",
                                   "BENIGN", "ATTACK")

    counts = df[label_col].value_counts()
    rare_classes = counts[counts < MIN_CLASS_COUNT].index.tolist()
    if rare_classes:
        print(f"Grouping {len(rare_classes)} rare attack classes "
              f"(<{MIN_CLASS_COUNT} samples) into 'Other': {rare_classes}")
    df["label_multiclass"] = df[label_col].apply(
        lambda x: "Other" if x in rare_classes else x
    )

    return df


def split_and_scale(
    df: pd.DataFrame,
    target: str = "label_binary",
    label_col: str = "Label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    """
    Separate features/labels, drop label-related columns from X,
    do a stratified train/test split, and standard-scale numeric
    features (fit on train only, applied to both).
    """
    label_cols = [label_col, "label_binary", "label_multiclass"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    return PreparedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        label_classes=sorted(y.unique().tolist()),
        scaler=scaler,
    )
