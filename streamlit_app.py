"""Interactive BEST-performer risk explorer for the XGBoost model.

Authors: Marc-Anthony Chouillard, Clément Pastier
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# Columns excluded during model training
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

# Configuration of the user interface for each feature
FEATURE_DEFINITIONS: Dict[str, List[Dict[str, Any]]] = {
    "Clinical background": [
        {
            "column": "Center_type",
            "label": "Center type",
            "type": "select",
            "options": ["CHG", "CHU", "ESPIC", "LIB"],
            "default": "CHU",
            "help": "Institution category"
        },
        {
            "column": "Age",
            "label": "Age",
            "type": "slider",
            "min": 18,
            "max": 90,
            "default": 60,
            "step": 1,
            "help": "Patient age in years"
        },
        {
            "column": "Men",
            "label": "Male",
            "type": "checkbox",
            "default": False,
            "help": "Tick for male patients"
        },
        {
            "column": "BMI",
            "label": "BMI",
            "type": "slider",
            "min": 15.0,
            "max": 50.0,
            "default": 25.0,
            "step": 0.1,
            "help": "Body Mass Index"
        },
        {
            "column": "ASA",
            "label": "ASA",
            "type": "radio",
            "options": ["1", "2", "3", "4"],
            "default": "2",
            "help": "ASA classification"
        },
        {
            "column": "Dependant_OMS",
            "label": "Dependent OMS",
            "type": "checkbox",
            "default": False,
            "help": "Dependent functional status"
        },
        {
            "column": "Laparotomy_history",
            "label": "Prior laparotomy",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Past_surgical_history",
            "label": "Past surgical history",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "No_antiplatelet_agent",
            "label": "No antiplatelet agent",
            "type": "checkbox",
            "default": True
        },
        {
            "column": "No_anticoagulant_agent",
            "label": "No anticoagulant agent",
            "type": "checkbox",
            "default": True
        },
        {
            "column": "Steroids",
            "label": "Chronic steroids",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Diabetes_mellitus",
            "label": "Diabetes",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Chronic_kidney_disease",
            "label": "Chronic kidney disease",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Cardiac_history",
            "label": "Cardiac history",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Arterial_hypertension",
            "label": "Arterial hypertension",
            "type": "checkbox",
            "default": True
        },
        {
            "column": "Chronic_obstructive_pulmonary_disease",
            "label": "COPD",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Hepatocellular_insufficiency",
            "label": "Hepatic insufficiency",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Chronic_calcifying_pancreatitis",
            "label": "Chronic calcifying pancreatitis",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Clinical_presentation",
            "label": "Clinical presentation",
            "type": "select",
            "options": [
                "diabetes", "diarrhea", "fever", "haemorrhage", "hypoglycaemia",
                "incidentaloma", "jaundice", "pain", "pancreatitis",
                "poor general condition", "pulmonary embolism", "trauma"
            ],
            "default": "incidentaloma"
        },
        {
            "column": "Neoadjuvant_treatment",
            "label": "Neoadjuvant treatment",
            "type": "select",
            "options": ["none", "chemotherapy", "chemoradiotherapy"],
            "default": "none"
        },
        {
            "column": "pancreatic_texture",
            "label": "Pancreatic texture",
            "type": "radio",
            "options": ["hard", "soft"],
            "default": "hard"
        },
        {
            "column": "pancreatic_texture_missing",
            "label": "Pancreatic texture missing",
            "type": "checkbox",
            "default": False,
            "help": "Tick if pancreatic texture is unknown"
        }
    ],
    "Histology and staging": [
        {
            "column": "Anatomical_pathology",
            "label": "Anatomical pathology",
            "type": "select",
            "options": [
                "auto immune pancreatitis", "carcinoma", "chronic pancreatitis",
                "ductal adenocarcinoma", "IPMN", "metastasis", "mucinous cystadenoma",
                "neuro endocrine tumor", "none", "other", "pseudocyst",
                "serous cystadenoma", "solid pseudopapillary neoplasm"
            ],
            "default": "neuro endocrine tumor"
        },
        {
            "column": "Tumor_size_mm",
            "label": "Tumor size (mm)",
            "type": "slider",
            "min": 0,
            "max": 150,
            "default": 35,
            "step": 1
        },
        {
            "column": "Preoperative_biopsy",
            "label": "Preoperative biopsy",
            "type": "select",
            "options": ["none", "percutaneous biopsy", "EUS-guided biopsy"],
            "default": "none"
        },
        {
            "column": "Center_expertise",
            "label": "Center expertise",
            "type": "select",
            "options": ["low", "intermediate", "high"],
            "default": "high"
        },
        {
            "column": "Center_MIP_mean",
            "label": "Center MIP mean",
            "type": "slider",
            "min": 0,
            "max": 40,
            "default": 10,
            "step": 1,
            "help": "Average minimally invasive pancreatectomies per year"
        },
        {
            "column": "Pancreatic_section_level",
            "label": "Pancreatic section level",
            "type": "select",
            "options": ["head", "neck", "body tail"],
            "default": "neck"
        }
    ],
    "Operative plan": [
        {
            "column": "Panned_splenectomy",
            "label": "Planned splenectomy",
            "type": "checkbox",
            "default": True
        },
        {
            "column": "LDP_modalities ",
            "label": "LDP modalities",
            "type": "select",
            "options": ["Warshaw", "Kimura", "distal splenopancreatectomy"],
            "default": "distal splenopancreatectomy"
        },
        {
            "column": "Robotic_assisted",
            "label": "Robotic assisted",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Hand_assisted_procedure",
            "label": "Hand assisted",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Conversion_to_laparotomy",
            "label": "Conversion to laparotomy",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Pancreatic_approach",
            "label": "Pancreatic approach",
            "type": "select",
            "options": ["gastrocolic ligament dissection", "coloepiploic dissection"],
            "default": "gastrocolic ligament dissection"
        },
        {
            "column": "Gastric_hanging",
            "label": "Gastric hanging",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Specimen_extraction_scar",
            "label": "Specimen extraction scar",
            "type": "select",
            "options": ["pfannenstiel", "trocar site enlargement", "other laparotomy"],
            "default": "pfannenstiel"
        },
        {
            "column": "RAMPS",
            "label": "RAMPS",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Mesentericoportal_axis_resection",
            "label": "Mesentericoportal axis resection",
            "type": "checkbox",
            "default": False
        }
    ],
    "Intraoperative course": [
        {
            "column": "Operative_duration",
            "label": "Operative duration (min)",
            "type": "slider",
            "min": 60,
            "max": 420,
            "default": 210,
            "step": 5
        },
        {
            "column": "Blood_loss",
            "label": "Blood loss (mL)",
            "type": "slider",
            "min": 0,
            "max": 1500,
            "default": 200,
            "step": 10
        },
        {
            "column": "Blood_loss_missing",
            "label": "Blood loss missing",
            "type": "checkbox",
            "default": False,
            "help": "Tick if intraoperative blood loss is unknown"
        },
        {
            "column": "Blood_transfusion",
            "label": "Blood transfusion (units)",
            "type": "slider",
            "min": 0,
            "max": 10,
            "default": 0,
            "step": 1
        },
        {
            "column": "Blood_transfusion_missing",
            "label": "Blood transfusion missing",
            "type": "checkbox",
            "default": False,
            "help": "Tick if transfusion data is unavailable"
        },
        {
            "column": "Drainage_modalities",
            "label": "Drainage modality",
            "type": "select",
            "options": ["active", "passive"],
            "default": "passive"
        },
        {
            "column": "Cholecystectomy",
            "label": "Cholecystectomy",
            "type": "checkbox",
            "default": False
        },
        {
            "column": "Multivisceral_resection",
            "label": "Multivisceral resection",
            "type": "checkbox",
            "default": False
        }
    ]
}
*** End Patch


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

    X = X.apply(pd.to_numeric, errors="coerce")

    return X, derived_indicators


def render_widget(feature: Dict[str, Any]):
    label = feature["label"]
    kind = feature["type"]
    help_text = feature.get("help")

    if kind == "slider":
        return st.slider(
            label,
            min_value=feature["min"],
            max_value=feature["max"],
            value=feature["default"],
            step=feature["step"],
            help=help_text
        )
    if kind == "checkbox":
        return st.checkbox(label, value=feature["default"], help=help_text)
    if kind == "select":
        options = feature["options"]
        index = options.index(feature["default"])
        return st.selectbox(label, options=options, index=index, help=help_text)
    if kind == "radio":
        options = feature["options"]
        index = options.index(feature["default"])
        return st.radio(label, options=options, index=index, help=help_text)
    return st.text_input(label, value=feature.get("default", ""), help=help_text)


def collect_user_inputs() -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    for section, features in FEATURE_DEFINITIONS.items():
        st.subheader(section)
        columns = st.columns(min(3, len(features)))
        for idx, feature in enumerate(features):
            column_slot = columns[idx % len(columns)]
            with column_slot:
                inputs[feature["column"]] = render_widget(feature)
    return inputs


def preprocess_user_inputs(inputs: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    record: Dict[str, Any] = {}
    for column, value in inputs.items():
        record[column] = value

    # Handle explicit missing indicators
    missing_pairs = [
        ("pancreatic_texture_missing", "pancreatic_texture"),
        ("Blood_loss_missing", "Blood_loss"),
        ("Blood_transfusion_missing", "Blood_transfusion"),
    ]
    for indicator, base in missing_pairs:
        flag = record.get(indicator, 0)
        flag_numeric = 1.0 if isinstance(flag, bool) and flag else float(flag)
        if flag_numeric >= 1.0:
            record[indicator] = 1.0
            record[base] = np.nan
        else:
            record[indicator] = 0.0

    for column, value in list(record.items()):
        if isinstance(value, bool):
            record[column] = 1.0 if value else 0.0

    df = pd.DataFrame([record])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


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
) -> Tuple[np.ndarray, np.ndarray]:
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
    calibrated = calibrator.transform(probabilities) if calibrator is not None else probabilities
    return probabilities, calibrated


def render_predictions(raw: np.ndarray, calibrated: np.ndarray) -> None:
    st.success("Prediction complete")
    st.metric("Raw probability", f"{raw[0]:.1%}")
    st.metric("Calibrated probability", f"{calibrated[0]:.1%}")
    st.caption("Calibrated output applies the Platt scaling trained in the production pipeline.")


def main() -> None:
    st.set_page_config(page_title="BEST Performer Risk Explorer", layout="wide")
    st.title("BEST Performer Probability Explorer")
    st.write(
        "Provide clinical data below to estimate the probability of achieving a BEST postoperative performance."
    )

    artifacts = initialise_pipeline()
    model = artifacts["model"]
    calibrator = artifacts["calibrator"]
    imputer = artifacts["imputer"]
    model_columns = artifacts["model_columns"]
    mapping_dict = artifacts["mapping_dict"]
    indicator_columns = artifacts["indicator_columns"]
    feature_columns = artifacts["feature_columns"]

    inputs = collect_user_inputs()

    if st.button("Compute probability", type="primary"):
        user_df = preprocess_user_inputs(inputs, feature_columns)
        raw, calibrated = predict_probabilities(
            user_df,
            model=model,
            calibrator=calibrator,
            imputer=imputer,
            model_columns=model_columns,
            mapping_dict=mapping_dict,
            indicator_columns=indicator_columns,
        )
        render_predictions(raw, calibrated)

        with st.expander("Submitted values"):
            st.dataframe(user_df)

    st.sidebar.header("About")
    st.sidebar.markdown(
        "- Predictions reuse the exported production XGBoost model; no retraining occurs on the platform.\n"
        "- Toggle the missing indicators whenever information is unavailable—the app will set the corresponding value to missing before imputation.\n"
        "- Use the calibrated probability as decision support alongside clinical judgement."
    )


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


if __name__ == "__main__":  # pragma: no cover
    main()
