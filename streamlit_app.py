"""Interactive BEST-performer risk explorer for the XGBoost model.

Authors: Marc-Anthony Chouillard, Clément Pastier
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


EXCLUDE_COLS: List[str] = [
    "Optimal_ideal_outcome",
    "Best_performers",
    "ISGPS_Grade",
    "Any_Hemorrhage",
    "Any_Bile_Leakage",
    "Any_POPF",
    "Re_operation_anycause",
    "Readmission",
    "Mortality",
    "LOS",
    "Any_SSI",
    "Any_organ_SSI",
    "Any_POPF_Bile_SSI",
]

MISSING_SUFFIX = "_missing"


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[object, object | None, pd.DataFrame, Dict[str, List[str]], pd.DataFrame]:
    """Load trained model, metadata, and training dataset."""

    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "results" / "production_imputed"

    model_path = artifacts_dir / "models" / "final_model.pkl"
    calibrator_path = artifacts_dir / "models" / "final_calibrator.pkl"
    mappings_path = artifacts_dir / "results" / "feature_mappings.csv"
    feature_types_path = artifacts_dir / "results" / "feature_types.json"
    training_path = base_dir / "base4.csv"

    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
    feature_mappings = pd.read_csv(mappings_path)
    with open(feature_types_path, "r", encoding="utf-8") as fh:
        feature_types = json.load(fh)
    training_df = pd.read_csv(training_path)

    return model, calibrator, feature_mappings, feature_types, training_df


def build_mapping_dict(feature_mappings: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    mapping: Dict[str, Dict[str, float]] = {}
    for feat, group in feature_mappings.groupby("Feature"):
        mapping[feat] = {
            row.Original_Label: float(row.Encoded_Value)
            for row in group.itertuples()
            if pd.notna(row.Original_Label)
        }
    return mapping


def determine_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in EXCLUDE_COLS]


def prepare_features(
    df: pd.DataFrame,
    feature_mappings: Dict[str, Dict[str, float]],
    missing_threshold: float,
    model_columns: Iterable[str] | None,
    indicator_columns: Iterable[str] | None,
) -> Tuple[pd.DataFrame, List[str]]:
    columns = determine_feature_columns(df)
    X = df[columns].copy()

    # Categorical encoding using stored mappings
    for col in X.columns:
        if col in feature_mappings:
            mapping = feature_mappings[col]
            X[col] = X[col].map(mapping)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    derived_indicators: List[str] = []
    if indicator_columns is None:
        for col in list(X.columns):
            missing_rate = X[col].isna().mean()
            if missing_rate > missing_threshold:
                indicator = f"{col}{MISSING_SUFFIX}"
                X[indicator] = X[col].isna().astype(float)
                derived_indicators.append(indicator)
    else:
        for indicator in indicator_columns:
            base = indicator[:-len(MISSING_SUFFIX)]
            if base in X.columns:
                X[indicator] = X[base].isna().astype(float)
            else:
                X[indicator] = 0.0

    X = X.replace(-1, np.nan)

    if model_columns is not None:
        for col in model_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X.loc[:, list(model_columns)]

    # Ensure numeric dtype
    X = X.apply(pd.to_numeric, errors="coerce")

    return X, derived_indicators


@st.cache_resource(show_spinner=False)
def initialise_pipeline():
    model, calibrator, feature_mapping_df, feature_types, training_df = load_artifacts()

    mapping_dict = build_mapping_dict(feature_mapping_df)
    X_train, indicator_columns = prepare_features(
        training_df,
        feature_mappings=mapping_dict,
        missing_threshold=0.10,
        model_columns=None,
        indicator_columns=None,
    )

    imputer = IterativeImputer(
        random_state=42,
        max_iter=10,
        initial_strategy="median",
        imputation_order="ascending",
    )
    imputer.fit(X_train)

    model_columns = X_train.columns.tolist()
    feature_columns = determine_feature_columns(training_df)

    return {
        "model": model,
        "calibrator": calibrator,
        "imputer": imputer,
        "model_columns": model_columns,
        "mapping_dict": mapping_dict,
        "indicator_columns": indicator_columns,
        "feature_types": feature_types,
        "training_df": training_df,
        "feature_columns": feature_columns,
    }


def predict_probabilities(
    df: pd.DataFrame,
    *,
    model,
    calibrator,
    imputer,
    model_columns: List[str],
    mapping_dict: Dict[str, Dict[str, float]],
    indicator_columns: Iterable[str],
    missing_threshold: float = 0.10,
) -> np.ndarray:
    X_prepared, _ = prepare_features(
        df,
        feature_mappings=mapping_dict,
        missing_threshold=missing_threshold,
        model_columns=model_columns,
        indicator_columns=indicator_columns,
    )

    X_imputed = imputer.transform(X_prepared)
    X_imputed_df = pd.DataFrame(X_imputed, columns=model_columns)

    probabilities = model.predict_proba(X_imputed_df)[:, 1]
    if calibrator is not None:
        calibrated = calibrator.transform(probabilities)
    else:
        calibrated = probabilities
    return probabilities, calibrated


def render_patient_editor(training_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    st.markdown("### Edit patient profile")
    default_idx = int(st.number_input("Row index", min_value=0, max_value=len(training_df) - 1, value=0))
    sample = training_df.loc[[default_idx], feature_columns]
    edited = st.data_editor(sample, num_rows="dynamic", key="patient_editor")
    return edited


def render_predictions(raw: np.ndarray, calibrated: np.ndarray, df: pd.DataFrame) -> None:
    st.success("Prediction complete")
    for idx, (prob, cal_prob) in enumerate(zip(raw, calibrated)):
        st.metric(
            label=f"Entry {idx + 1}",
            value=f"Raw probability: {prob:.2%}",
            delta=f"Calibrated: {cal_prob:.2%}"
        )
    st.caption("Δ values reflect isotonic-calibrated probabilities derived from the production pipeline.")
    with st.expander("Preview of submitted data"):
        st.dataframe(df, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="BEST Performer Risk Explorer", layout="wide")
    st.title("BEST Performer Probability Explorer")
    st.write(
        "Upload a patient profile or edit a row from the training dataset to estimate the probability "
        "of achieving a BEST postoperative outcome."
    )

    artifacts = initialise_pipeline()
    model = artifacts["model"]
    calibrator = artifacts["calibrator"]
    imputer = artifacts["imputer"]
    model_columns = artifacts["model_columns"]
    mapping_dict = artifacts["mapping_dict"]
    indicator_columns = artifacts["indicator_columns"]
    training_df = artifacts["training_df"]
    feature_columns = artifacts["feature_columns"]

    tab_editor, tab_upload = st.tabs(["Interactive editor", "Batch upload"])

    with tab_editor:
        edited_df = render_patient_editor(training_df, feature_columns)
        if st.button("Predict from edited row", key="predict_editor"):
            if not edited_df.empty:
                probabilities, calibrated = predict_probabilities(
                    edited_df,
                    model=model,
                    calibrator=calibrator,
                    imputer=imputer,
                    model_columns=model_columns,
                    mapping_dict=mapping_dict,
                    indicator_columns=indicator_columns,
                )
                render_predictions(probabilities, calibrated, edited_df)
            else:
                st.error("No data available for prediction.")

    with tab_upload:
        uploader = st.file_uploader("Upload CSV", type="csv")
        if uploader is not None:
            try:
                uploaded_df = pd.read_csv(uploader)
            except Exception as exc:  # pragma: no cover - user supplied file
                st.error(f"Failed to read CSV: {exc}")
            else:
                st.write("Preview", uploaded_df.head())
                if st.button("Predict from uploaded data", key="predict_upload"):
                    probabilities, calibrated = predict_probabilities(
                        uploaded_df,
                        model=model,
                        calibrator=calibrator,
                        imputer=imputer,
                        model_columns=model_columns,
                        mapping_dict=mapping_dict,
                        indicator_columns=indicator_columns,
                    )
                    render_predictions(probabilities, calibrated, uploaded_df)

    st.sidebar.header("Usage")
    st.sidebar.markdown(
        "- Edit a training row or upload a CSV containing the required features (see README).\n"
        "- All computations happen in-memory; no data is persisted.\n"
        "- Use the exported probability as a decision-support aid only—clinical validation remains essential."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
