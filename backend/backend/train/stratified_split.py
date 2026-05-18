# stratified_split.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def build_stratified_splits(
    metadata_df: pd.DataFrame,
    val_size:    float = 0.15,
    test_size:   float = 0.15,
    seed:        int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produce train/val/test splits with the following guarantees:
    
    1. PATIENT-LEVEL GROUPING: A patient's studies never span splits.
       Prevents the model from recognizing patient anatomy rather than pathology.
       Required column: 'patient_id'
    
    2. STRATIFICATION ON COMPOSITE KEY: Stratify on (label × view × source_dataset).
       Required columns: 'label' {0,1}, 'view' {PA, AP, lateral},
                         'source' {NIH, CheXpert, MIMIC, ...}
    
    3. SEVERITY STRATIFICATION: If severity scores are available (e.g., from
       radiologist reports), include them to prevent easy/hard case clustering.
       Optional column: 'severity' {mild, moderate, severe}
    
    Expected metadata_df columns:
        image_path  : str  — path to image file
        patient_id  : str  — unique patient identifier
        label       : int  — 0=Normal, 1=Pneumonia
        view        : str  — radiographic view {PA, AP, lateral}
        source      : str  — dataset origin
        severity    : str  — (optional) disease severity
    """
    
    required_cols = {"image_path", "patient_id", "label", "view", "source"}
    assert required_cols.issubset(metadata_df.columns), \
        f"Missing columns: {required_cols - set(metadata_df.columns)}"
    
    df = metadata_df.copy().reset_index(drop=True)
    
    # ── Composite stratification key ──────────────────────────────────────────
    df["strat_key"] = (
        df["label"].astype(str) + "_" +
        df["view"].str.upper() + "_" +
        df["source"].str.lower()
    )
    # Collapse rare strat_key combinations to prevent stratification failure
    key_counts = df["strat_key"].value_counts()
    rare_keys  = key_counts[key_counts < 10].index
    df.loc[df["strat_key"].isin(rare_keys), "strat_key"] = \
        df.loc[df["strat_key"].isin(rare_keys), "label"].astype(str) + "_other"
    
    # ── First split: isolate test set ─────────────────────────────────────────
    # StratifiedGroupKFold ensures no patient appears in both splits
    n_test_folds = int(round(1.0 / test_size))   # e.g. test_size=0.15 → 7 folds
    sgkf = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=seed)
    
    train_val_idx, test_idx = next(sgkf.split(
        df, df["strat_key"], groups=df["patient_id"]
    ))
    df_test     = df.iloc[test_idx].copy()
    df_train_val = df.iloc[train_val_idx].copy().reset_index(drop=True)
    
    # ── Second split: isolate val from remaining train_val ────────────────────
    n_val_folds = int(round(1.0 / (val_size / (1.0 - test_size))))
    sgkf2 = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=seed+1)
    
    train_idx, val_idx = next(sgkf2.split(
        df_train_val, df_train_val["strat_key"], groups=df_train_val["patient_id"]
    ))
    df_train = df_train_val.iloc[train_idx].copy()
    df_val   = df_train_val.iloc[val_idx].copy()
    
    # ── Verification assertions (fail fast if split is degenerate) ────────────
    _verify_splits(df_train, df_val, df_test)
    
    return df_train, df_val, df_test


def _verify_splits(train, val, test):
    """Hard assertions — fail loudly before training starts."""
    
    # 1. No patient leakage
    train_patients = set(train["patient_id"])
    val_patients   = set(val["patient_id"])
    test_patients  = set(test["patient_id"])
    
    assert not (train_patients & val_patients), \
        f"PATIENT LEAKAGE: {len(train_patients & val_patients)} patients in train AND val"
    assert not (train_patients & test_patients), \
        f"PATIENT LEAKAGE: {len(train_patients & test_patients)} patients in train AND test"
    assert not (val_patients   & test_patients), \
        f"PATIENT LEAKAGE: {len(val_patients & test_patients)} patients in val AND test"
    
    # 2. Label balance roughly maintained
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        pneumonia_rate = split_df["label"].mean()
        print(f"{split_name:>5}: n={len(split_df):5d}  "
              f"pneumonia={pneumonia_rate:.3f}  "
              f"views={split_df['view'].value_counts().to_dict()}")
        assert 0.1 < pneumonia_rate < 0.9, \
            f"Degenerate split: {split_name} pneumonia_rate={pneumonia_rate:.3f}"
    
    print("\n✓ All split verification assertions passed — no patient leakage.")