"""
data_loader.py

Loads and concatenates the CICIDS2017 CSV files (CICFlowMeter output).

The official dataset ships as 8 separate CSV files, one per
day/scenario, found here:
    https://www.unb.ca/cic/datasets/ids-2017.html

Place the raw CSVs (the "MachineLearningCSV" / "GeneratedLabelledFlows"
variant) in `data/raw/` before running this module.

Known quirks this module accounts for:
  - Column names have leading/trailing whitespace
    (e.g. " Label", " Destination Port")
  - 'Flow Bytes/s' and 'Flow Packets/s' can contain Infinity / -Infinity
  - Some rows are exact duplicates across files
  - A handful of rows have NaNs in numeric columns
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names so they're consistent across files."""
    df.columns = [c.strip() for c in df.columns]
    return df


def _fix_label_text(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """
    The 'Web Attack' labels in the raw CSVs contain a Windows-1252
    en-dash (0x96) which, when the file is read as latin-1, becomes
    a mojibake sequence (e.g. 'Web Attack ï¿½ Brute Force'). Normalize
    these to a plain hyphen so labels are clean and consistent.
    """
    if label_col in df.columns:
        df[label_col] = (
            df[label_col]
            .str.replace("ï¿½", "-", regex=False)     # UTF-8 en-dash misread as latin-1 (3 chars)
            .str.replace("\ufffd", "-", regex=False)  # U+FFFD replacement char
            .str.replace("\x96", "-", regex=False)
            .str.replace("Web Attack -", "Web Attack -", regex=False)
            .str.strip()
        )
    return df


def load_cicids2017(raw_dir: str | Path = RAW_DIR, verbose: bool = True) -> pd.DataFrame:
    """
    Load every CSV in `raw_dir`, clean column names, and concatenate
    into a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Combined raw dataframe (still needs cleaning via
        `preprocessing.clean_dataframe`).
    """
    raw_dir = Path(raw_dir)
    csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}.\n"
            "Download the CICIDS2017 'MachineLearningCSV' files from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and place "
            "them in this folder."
        )

    frames = []
    for f in csv_files:
        if verbose:
            print(f"Loading {os.path.basename(f)} ...")
        df = pd.read_csv(f, low_memory=False, encoding="latin1")
        df = _clean_columns(df)
        df = _fix_label_text(df)
        df["source_file"] = os.path.basename(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if verbose:
        print(f"\nCombined shape: {combined.shape}")
        print(f"Files loaded: {len(csv_files)}")

    return combined


if __name__ == "__main__":
    df = load_cicids2017()
    print(df["Label"].value_counts())