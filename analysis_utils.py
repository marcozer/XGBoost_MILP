"""Shared helper utilities for comparative analyses.

Authors: Marc-Anthony Chouillard, Clément Pastier
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# Feature definitions mirroring generate_logistic_predictions() in 58_final_imputed.py
LOGISTIC_CONTINUOUS = ['Age', 'BMI', 'Center_MIP_mean', 'ASA', 'Operative_duration', 'Blood_loss']
LOGISTIC_BINARY = ['Men', 'Conversion_to_laparotomy', 'Panned_splenectomy']


def format_p_value(p_value: float) -> str:
    """Return a human-friendly representation of a p-value."""

    if p_value is None or np.isnan(p_value):
        return "N/A"
    if p_value >= 1e-3:
        return f"{p_value:.3f}"

    mantissa, exponent = f"{p_value:.2e}".split("e")
    exponent = int(exponent)
    return f"{float(mantissa):.2f} x 10^{exponent}"


def _prepare_logistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the feature matrix exactly as done in 58_final_imputed.py."""

    X = pd.DataFrame(index=df.index)

    for var in LOGISTIC_CONTINUOUS:
        if var in df.columns:
            X[var] = df[var].astype(float)
        else:
            X[var] = 0.0

    for var in LOGISTIC_BINARY:
        if var in df.columns:
            X[var] = df[var].astype(float)
        else:
            X[var] = 0.0

    if 'Center_expertise' in df.columns:
        if df['Center_expertise'].dtype == object:
            expertise_map = {'Low': 0, 'Intermediate': 1, 'High': 2}
            X['Center_expertise'] = df['Center_expertise'].map(expertise_map)
        else:
            X['Center_expertise'] = df['Center_expertise'].astype(float)
    else:
        X['Center_expertise'] = 0.0

    return X[LOGISTIC_CONTINUOUS + LOGISTIC_BINARY + ['Center_expertise']]


def generate_logistic_predictions(base_dir: Path, *, random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the logistic baseline predictions used in the 58-script pipeline."""

    df = pd.read_csv(base_dir / 'base_marc_anthony_05082025.csv')
    if 'Best_performers' not in df.columns:
        raise ValueError("Expected 'Best_performers' column in dataset for logistic baseline.")

    y = df['Best_performers'].values
    X = _prepare_logistic_features(df)

    valid_mask = ~pd.isna(y)
    X_valid = X.loc[valid_mask].reset_index(drop=True)
    y_valid = y[valid_mask].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    probabilities = np.zeros(len(y_valid))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_valid, y_valid), start=1):
        X_train = X_valid.iloc[train_idx].copy()
        X_test = X_valid.iloc[test_idx].copy()
        y_train = y_valid[train_idx]

        imputer = IterativeImputer(max_iter=10, random_state=random_state + fold, verbose=0)
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        probabilities[test_idx] = model.predict_proba(X_test_scaled)[:, 1]

    return y_valid, probabilities
