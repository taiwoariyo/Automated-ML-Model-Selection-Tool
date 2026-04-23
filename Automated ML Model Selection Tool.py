#pip install --upgrade pip
#streamlit run automated_ml_tool.py
#pip install openpyxl

# ============================================================
# File: automated_ml_tool.py
# Purpose:
# A polished, public-facing Streamlit AutoML application that:
# - loads CSV / Excel / JSON files
# - cleans and profiles messy real-world datasets
# - auto-detects classification vs regression
# - compares multiple ML models
# - evaluates the best model on a holdout test set
# - allows optional hyperparameter tuning
# - exports the trained pipeline for reuse
#
# Notes:
# - This version emphasizes readability, strong comments,
#   professional UI, and more robust data handling.
# ============================================================

# -----------------------------
# Standard library imports
# -----------------------------
import io
import warnings
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Third-party imports
# -----------------------------
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Silence noisy warnings to keep the public-facing experience cleaner
warnings.filterwarnings("ignore")

# -----------------------------
# Optional gradient boosting libraries
# The app still runs even if these are not installed
# -----------------------------
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="AutoML Model Selection Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more polished, launch-ready interface
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0A1128 0%, #001F54 40%, #034078 70%, #1282A2 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(46,120,255,0.18), rgba(0,191,165,0.12));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.4rem 1.4rem 1.1rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.18);
        }

        .section-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px;
            padding: 1rem 1rem 0.8rem 1rem;
            margin-bottom: 1rem;
        }

        .tiny-note {
            color: #b7c6d9;
            font-size: 0.92rem;
        }

        h1, h2, h3 {
            letter-spacing: 0.2px;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 0.85rem;
            border-radius: 14px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        .stButton > button {
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 600;
        }

        .stDownloadButton > button {
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS: DATA LOADING / CLEANING
# ============================================================
def make_column_names_unique(columns: List[str]) -> List[str]:
    """
    Clean and make column names unique.

    Real datasets often contain duplicate headers, blank headers,
    or headers with trailing spaces/newlines.
    """
    counts = {}
    new_columns = []

    for col in columns:
        clean_col = str(col).strip().replace("\n", " ")

        if clean_col == "" or clean_col.lower() in {"nan", "none", "null"}:
            clean_col = "Unnamed"

        if clean_col not in counts:
            counts[clean_col] = 0
            new_columns.append(clean_col)
        else:
            counts[clean_col] += 1
            new_columns.append(f"{clean_col}_{counts[clean_col]}")

    return new_columns


def score_header_row(row: pd.Series) -> float:
    """
    Assign a heuristic score to a row to decide whether it looks like a header.
    """
    row_str = row.fillna("").astype(str).str.strip()

    if len(row_str) == 0:
        return -999

    bad_tokens = {"", "nan", "null", "none", "unnamed"}
    bad_ratio = row_str.str.lower().isin(bad_tokens).mean()
    unique_ratio = row_str.nunique(dropna=False) / max(len(row_str), 1)
    alpha_ratio = row_str.apply(lambda x: any(ch.isalpha() for ch in x)).mean()

    score = (alpha_ratio * 0.5) + (unique_ratio * 0.35) - (bad_ratio * 0.9)
    return score


def detect_header_row(raw_data: pd.DataFrame, max_rows_to_check: int = 8) -> int:
    """
    Detect which row most likely contains the headers.

    This helps when files contain title rows, report captions,
    or metadata lines above the actual dataset.
    """
    rows_to_check = min(max_rows_to_check, len(raw_data))
    if rows_to_check == 0:
        return 0

    best_row = 0
    best_score = -999

    for i in range(rows_to_check):
        row_score = score_header_row(raw_data.iloc[i])
        if row_score > best_score:
            best_score = row_score
            best_row = i

    return best_row



def try_read_csv(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV reader with fallback encodings.
    """
    uploaded_file.seek(0)
    try:
        return pd.read_csv(uploaded_file, header=None)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, header=None, encoding="latin1")



def try_read_excel(uploaded_file) -> pd.DataFrame:
    """
    Robust Excel reader.

    Supports xlsx / xls / xlsm / xlsb more gracefully,
    reads the first sheet by default,
    and falls back across engines when possible.
    """
    filename = uploaded_file.name.lower()
    uploaded_file.seek(0)

    engine_candidates = []
    if filename.endswith(".xlsx") or filename.endswith(".xlsm"):
        engine_candidates = ["openpyxl"]
    elif filename.endswith(".xls"):
        engine_candidates = ["xlrd", "openpyxl"]
    elif filename.endswith(".xlsb"):
        engine_candidates = ["pyxlsb", "openpyxl"]
    else:
        engine_candidates = [None, "openpyxl", "xlrd", "pyxlsb"]

    last_error = None
    for engine in engine_candidates:
        try:
            uploaded_file.seek(0)
            if engine is None:
                return pd.read_excel(uploaded_file, header=None)
            return pd.read_excel(uploaded_file, engine=engine, header=None)
        except Exception as e:
            last_error = e
            continue

    raise ValueError(
        "Excel file could not be read. Make sure the file is a valid Excel file and that the required engine is installed. "
        f"Last error: {last_error}"
    )



def try_read_json(uploaded_file) -> pd.DataFrame:
    """
    Robust JSON reader.

    Handles normal JSON files, record-style JSON, and line-delimited JSON.
    """
    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()

    try:
        text = raw_bytes.decode("utf-8")
    except Exception:
        text = raw_bytes.decode("latin1")

    try:
        return pd.read_json(io.StringIO(text))
    except Exception:
        pass

    try:
        return pd.read_json(io.StringIO(text), lines=True)
    except Exception:
        pass

    import json
    parsed = json.loads(text)

    if isinstance(parsed, list):
        return pd.json_normalize(parsed)
    if isinstance(parsed, dict):
        return pd.json_normalize(parsed)

    return pd.DataFrame(parsed)



def maybe_use_first_row_as_header(data: pd.DataFrame) -> bool:
    """
    Decide whether the first row of a DataFrame likely contains headers.
    """
    if data.empty:
        return False

    generic_header = all(str(col).isdigit() for col in data.columns)
    return generic_header



def clean_numeric_like_series(series: pd.Series) -> pd.Series:
    """
    Clean a potentially numeric text column before conversion.

    Handles:
    - commas: 1,200
    - currency: $1200
    - percentages: 45%
    - negatives in parentheses: (500)
    - spaces
    - common null tokens
    """
    cleaned = series.copy()
    cleaned = cleaned.astype(str).str.strip()

    cleaned = cleaned.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan,
            "none": np.nan,
            "null": np.nan,
            "NULL": np.nan,
            "N/A": np.nan,
            "n/a": np.nan,
            "NA": np.nan,
        }
    )

    cleaned = cleaned.str.replace(r"^\((.+)\)$", r"-\1", regex=True)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("$", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)

    return pd.to_numeric(cleaned, errors="coerce")



def load_and_clean_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load the uploaded dataset and apply first-pass cleaning.
    """
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            raw_data = try_read_csv(uploaded_file)
            use_header_detection = True
        elif filename.endswith((".xlsx", ".xls", ".xlsm", ".xlsb")):
            raw_data = try_read_excel(uploaded_file)
            use_header_detection = True
        elif filename.endswith(".json"):
            raw_data = try_read_json(uploaded_file)
            use_header_detection = maybe_use_first_row_as_header(raw_data)
        else:
            raise ValueError("Unsupported file format. Please upload CSV, XLSX, XLS, XLSM, XLSB, or JSON.")

        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        if raw_data.empty:
            raise ValueError("The uploaded file appears to be empty.")

        if use_header_detection:
            header_row = detect_header_row(raw_data)
            data = raw_data.iloc[header_row + 1:].copy()
            detected_columns = raw_data.iloc[header_row].tolist()
            data.columns = make_column_names_unique(detected_columns)
        else:
            data = raw_data.copy()
            data.columns = make_column_names_unique(list(data.columns))

        data = data.replace(r"^\s*$", np.nan, regex=True)
        data = data.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

        for col in data.columns:
            if data[col].dtype == "object":
                data[col] = data[col].astype(str).str.strip()
                data.loc[data[col].str.lower().isin(["nan", "none", "null", ""]), col] = np.nan

        for col in data.columns:
            if data[col].dtype == "object":
                numeric_candidate = clean_numeric_like_series(data[col])
                non_null_original = data[col].notna().sum()
                if non_null_original > 0:
                    conversion_ratio = numeric_candidate.notna().sum() / non_null_original
                    if conversion_ratio >= 0.90:
                        data[col] = numeric_candidate

        return data

    except Exception as e:
        st.error(f"Failed to load the file: {e}")
        return None


# ============================================================
# HELPER FUNCTIONS: DATA PROFILING / TASK DETECTION
# ============================================================
def infer_feature_types(X: pd.DataFrame, numeric_threshold: float = 0.70):
    """
    Infer numeric vs categorical columns more reliably.
    """
    X = X.copy()
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        series = X[col]

        if pd.api.types.is_bool_dtype(series):
            X[col] = series.astype(str)
            categorical_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            numeric_cols.append(col)
            continue

        numeric_candidate = clean_numeric_like_series(series)
        non_null_original = series.notna().sum()
        non_null_converted = numeric_candidate.notna().sum()

        if non_null_original == 0:
            categorical_cols.append(col)
            continue

        converted_ratio = non_null_converted / non_null_original

        if converted_ratio >= numeric_threshold:
            X[col] = numeric_candidate
            numeric_cols.append(col)
        else:
            X[col] = series.astype(str)
            categorical_cols.append(col)

    return X, numeric_cols, categorical_cols



def get_dataset_profile(data: pd.DataFrame) -> Dict[str, float]:
    """
    Return a compact dataset quality profile for display.
    """
    total_cells = data.shape[0] * data.shape[1] if not data.empty else 0
    missing_cells = int(data.isna().sum().sum()) if total_cells > 0 else 0
    duplicate_rows = int(data.duplicated().sum()) if not data.empty else 0

    _, numeric_cols, categorical_cols = infer_feature_types(data.copy())

    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "missing_cells": missing_cells,
        "missing_pct": round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0,
        "duplicate_rows": duplicate_rows,
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(categorical_cols),
    }



def auto_detect_task(y_series: pd.Series) -> str:
    """
    Automatically infer whether the task is classification or regression.
    """
    y_non_null = y_series.dropna()

    if y_non_null.empty:
        return "classification"

    if pd.api.types.is_numeric_dtype(y_non_null):
        return "regression" if y_non_null.nunique() > max(15, len(y_non_null) * 0.05) else "classification"

    numeric_candidate = clean_numeric_like_series(y_non_null)
    conversion_ratio = numeric_candidate.notna().mean()

    if conversion_ratio >= 0.80:
        return "regression" if numeric_candidate.nunique() > max(15, len(numeric_candidate.dropna()) * 0.05) else "classification"

    return "classification"



def find_valid_targets(data: pd.DataFrame) -> List[str]:
    """
    Determine which columns are usable as target columns.
    """
    valid_targets = []

    for col in data.columns:
        col_lower = str(col).strip().lower()

        if col_lower.startswith("unnamed"):
            continue

        if data[col].notna().sum() < 10:
            continue

        if data[col].nunique(dropna=True) <= 1:
            continue

        valid_targets.append(col)

    return valid_targets


# ============================================================
# HELPER FUNCTIONS: FEATURE CLEANING / PREPROCESSING
# ============================================================
def convert_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to detect datetime-like columns and engineer useful parts.
    """
    X = X.copy()

    for col in X.columns:
        if X[col].dtype == "object":
            parsed = pd.to_datetime(X[col], errors="coerce")
            parse_ratio = parsed.notna().mean()

            if parse_ratio >= 0.7:
                X[f"{col}_year"] = parsed.dt.year
                X[f"{col}_month"] = parsed.dt.month
                X[f"{col}_day"] = parsed.dt.day
                X[f"{col}_dayofweek"] = parsed.dt.dayofweek
                X = X.drop(columns=[col])

    return X



def drop_problematic_columns(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Remove problematic feature columns and track what was removed.

    Removed columns include:
    - high-missing-value columns (> 60%)
    - constant columns
    - extremely high-cardinality text columns that often memorize IDs
    """
    X = X.copy()
    dropped_summary = {
        "high_missing": [],
        "constant": [],
        "high_cardinality": [],
    }

    high_missing_cols = X.columns[X.isna().mean() > 0.60].tolist()
    if high_missing_cols:
        dropped_summary["high_missing"] = high_missing_cols
        X = X.drop(columns=high_missing_cols)

    constant_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
    if constant_cols:
        dropped_summary["constant"] = constant_cols
        X = X.drop(columns=constant_cols)

    high_card_cols = []
    for col in X.columns:
        if X[col].dtype == "object":
            nunique = X[col].nunique(dropna=True)
            ratio = nunique / max(len(X), 1)
            if nunique > 100 and ratio > 0.80:
                high_card_cols.append(col)

    if high_card_cols:
        dropped_summary["high_cardinality"] = high_card_cols
        X = X.drop(columns=high_card_cols)

    return X, dropped_summary



def standardize_mixed_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize problematic mixed-type columns without destroying numeric information.
    """
    X = X.copy()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(str).str.strip()
            X.loc[X[col].str.lower().isin(["", "nan", "none", "null", "n/a"]), col] = np.nan

    return X



def build_preprocessor(X: pd.DataFrame):
    """
    Build a preprocessing pipeline for numeric and categorical columns.
    """
    X = X.copy()

    X = convert_datetime_features(X)
    X, dropped_summary = drop_problematic_columns(X)
    X = standardize_mixed_columns(X)
    X, numeric_cols, categorical_cols = infer_feature_types(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return X, preprocessor, numeric_cols, categorical_cols, dropped_summary


# ============================================================
# HELPER FUNCTIONS: METRICS / CV / MODELS / TUNING
# ============================================================
def is_imbalanced_classification(y) -> bool:
    """
    Determine whether a classification target is notably imbalanced.
    """
    y_series = pd.Series(y)
    if y_series.nunique() < 2:
        return False

    counts = y_series.value_counts(normalize=True)
    minority_share = counts.min()
    return minority_share < 0.20



def get_cv_strategy(y_train, task_type: str, preferred_folds: int = 5):
    """
    Return a safe and appropriate cross-validation splitter.
    """
    if task_type == "classification":
        class_counts = pd.Series(y_train).value_counts()
        min_class_count = int(class_counts.min()) if not class_counts.empty else 2
        folds = max(2, min(preferred_folds, min_class_count))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    folds = max(2, min(preferred_folds, len(y_train)))
    return KFold(n_splits=folds, shuffle=True, random_state=42)



def get_classification_primary_metric(y_train) -> str:
    """
    Use balanced accuracy on imbalanced targets and plain accuracy otherwise.
    """
    return "balanced_accuracy" if is_imbalanced_classification(y_train) else "accuracy"



def get_models(task_type: str) -> Dict[str, object]:
    """
    Return candidate models based on the task type.

    These are strong general-purpose baseline models.
    They improve your chances of finding a high-performing model,
    but they do NOT guarantee perfect accuracy on every dataset.
    """
    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=10000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
            "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
            "SVM": SVC(probability=False, class_weight="balanced"),
            "KNN": KNeighborsClassifier(),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
            )

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMClassifier(
                random_state=42,
                n_estimators=300,
                learning_rate=0.05,
            )

    else:
        models = {
            "Ridge": Ridge(),
            "ElasticNet": ElasticNet(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
            "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=42),
            "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
            "SVM": SVR(),
            "KNN": KNeighborsRegressor(),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(
                random_state=42,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
            )

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMRegressor(
                random_state=42,
                n_estimators=300,
                learning_rate=0.05,
            )

    return models



def build_pipeline(preprocessor, model) -> Pipeline:
    """
    Combine preprocessing and model into one reusable sklearn pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )



def get_param_grid(model_name: str, task_type: str) -> Dict[str, List]:
    """
    Hyperparameter search spaces for supported models.

    The grids are intentionally moderate to improve performance
    without making the Streamlit app too slow.
    """
    if model_name == "Random Forest":
        return {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }

    if model_name == "Extra Trees":
        return {
            "model__n_estimators": [200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }

    if model_name == "HistGradientBoosting":
        return {
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [None, 6, 12],
            "model__max_iter": [100, 200],
        }

    if model_name == "XGBoost":
        return {
            "model__n_estimators": [200, 300],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }

    if model_name == "LightGBM":
        return {
            "model__n_estimators": [200, 300],
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.03, 0.05, 0.1],
        }

    if model_name == "Logistic Regression" and task_type == "classification":
        return {
            "model__C": [0.1, 1.0, 3.0],
        }

    if model_name == "Ridge" and task_type == "regression":
        return {
            "model__alpha": [0.1, 1.0, 10.0],
        }

    if model_name == "ElasticNet" and task_type == "regression":
        return {
            "model__alpha": [0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        }

    if model_name == "KNN":
        return {
            "model__n_neighbors": [3, 5, 7, 11],
            "model__weights": ["uniform", "distance"],
        }

    if model_name == "SVM":
        return {
            "model__C": [0.5, 1.0, 3.0],
            "model__kernel": ["rbf", "linear"],
        }

    return {}



def run_grid_search(model_name: str, pipeline: Pipeline, X_train, y_train, task_type: str):
    """
    Run GridSearchCV on supported models.
    If the model is not supported or tuning fails, return the original pipeline.
    """
    param_grid = get_param_grid(model_name, task_type)

    if not param_grid:
        st.info(f"No tuning grid defined for {model_name}. Using default settings.")
        return pipeline, None

    try:
        cv_strategy = get_cv_strategy(y_train, task_type, preferred_folds=4)
        scoring = get_classification_primary_metric(y_train) if task_type == "classification" else "neg_root_mean_squared_error"

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        st.success(f"Best parameters for {model_name}: {grid.best_params_}")
        return grid.best_estimator_, grid.best_params_

    except Exception as e:
        st.warning(f"Grid search skipped for {model_name}: {e}")
        return pipeline, None



def evaluate_models(X_train, y_train, models: Dict[str, object], preprocessor, task_type: str) -> pd.DataFrame:
    """
    Evaluate candidate models using cross-validation.

    Key improvement:
    - classification is no longer judged only by plain accuracy
    - imbalanced problems prefer balanced accuracy
    - additional quality metrics are surfaced so poor models are less likely to mislead users
    """
    rows = []
    cv_strategy = get_cv_strategy(y_train, task_type, preferred_folds=5)

    for name, model in models.items():
        try:
            pipeline = build_pipeline(preprocessor, clone(model))

            if task_type == "classification":
                primary_metric = get_classification_primary_metric(y_train)
                scoring = {
                    "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy",
                    "f1_weighted": "f1_weighted",
                }
                cv_results = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_strategy,
                    scoring=scoring,
                    n_jobs=None,
                    error_score="raise",
                )

                rows.append(
                    {
                        "Model": name,
                        "Primary CV Score": float(np.mean(cv_results[f"test_{primary_metric}"])),
                        "Score Type": "Balanced Accuracy" if primary_metric == "balanced_accuracy" else "Accuracy",
                        "Accuracy": float(np.mean(cv_results["test_accuracy"])),
                        "Balanced Accuracy": float(np.mean(cv_results["test_balanced_accuracy"])),
                        "F1 Weighted": float(np.mean(cv_results["test_f1_weighted"])),
                    }
                )
            else:
                scoring = {
                    "rmse": "neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error",
                    "r2": "r2",
                }
                cv_results = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_strategy,
                    scoring=scoring,
                    n_jobs=None,
                    error_score="raise",
                )

                rows.append(
                    {
                        "Model": name,
                        "Primary CV Score": float(-np.mean(cv_results["test_rmse"])),
                        "Score Type": "RMSE",
                        "RMSE": float(-np.mean(cv_results["test_rmse"])),
                        "MAE": float(-np.mean(cv_results["test_mae"])),
                        "R²": float(np.mean(cv_results["test_r2"])),
                    }
                )

        except Exception:
            continue

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        ascending = task_type == "regression"
        results_df = results_df.sort_values("Primary CV Score", ascending=ascending).reset_index(drop=True)

    return results_df


# ============================================================
# HELPER FUNCTIONS: VISUALIZATION
# ============================================================
def plot_model_results(results_df: pd.DataFrame, task_type: str) -> None:
    """
    Display a clean single-plot bar chart comparing candidate models.
    """
    if results_df.empty:
        st.warning("No valid models could be evaluated.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(results_df["Model"], results_df["Primary CV Score"])
    ax.set_title("Model Comparison", fontsize=14)
    ax.set_xlabel("Model")

    if task_type == "classification":
        ax.set_ylabel(results_df.iloc[0]["Score Type"])
    else:
        ax.set_ylabel("CV RMSE")

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig)



def plot_target_distribution(y_raw: pd.Series, task_type: str) -> None:
    """
    Plot a simple target distribution to help users understand the modeling target.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    if task_type == "classification":
        value_counts = y_raw.astype(str).value_counts().head(15)
        ax.bar(value_counts.index.astype(str), value_counts.values)
        ax.set_title("Target Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        plt.xticks(rotation=25, ha="right")
    else:
        numeric_target = clean_numeric_like_series(y_raw).dropna()
        ax.hist(numeric_target, bins=25)
        ax.set_title("Target Distribution")
        ax.set_xlabel("Target Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    st.pyplot(fig)


# ============================================================
# HELPER FUNCTIONS: TRAIN/TEST SAFETY
# ============================================================
def safe_train_test_split(X, y, task_type: str):
    """
    Split data into train/test safely.
    """
    if task_type == "classification":
        try:
            y_series = pd.Series(y)
            min_class_count = y_series.value_counts().min()

            if min_class_count >= 2:
                return train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y,
                )
        except Exception:
            pass

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.3rem;">AutoML Model Selection Tool</h1>
            <p style="margin-bottom:0.4rem; font-size:1.04rem;">
                Upload a dataset, select your target, compare strong baseline models,
                and export the best trained pipeline.
            </p>
            <p class="tiny-note" style="margin-bottom:0;">
                Designed for a clean public demo experience while still handling messy real-world data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Workflow")
    st.sidebar.write("1. Upload data")
    st.sidebar.write("2. Choose target and features")
    st.sidebar.write("3. Run model comparison")
    st.sidebar.write("4. Review results and download the best pipeline")

    with st.sidebar.expander("Supported file types", expanded=False):
        st.write("CSV, XLSX, XLS, XLSM, XLSB, JSON")

    with st.sidebar.expander("What this app does", expanded=False):
        st.write(
            """
            - Cleans messy headers
            - Handles missing values
            - Detects numeric-like text columns
            - Distinguishes categorical vs numerical features more reliably
            - Encodes categorical columns
            - Compares several ML models more safely
            - Uses better metrics for imbalanced classification
            - Evaluates the best model on a holdout set
            - Exports the trained pipeline
            """
        )

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "xlsm", "xlsb", "json"],
    )

    if not uploaded_file:
        st.info("Upload a dataset to begin.")
        return

    data = load_and_clean_data(uploaded_file)

    if data is None or data.empty:
        st.error("The uploaded file could not be processed.")
        return

    profile = get_dataset_profile(data)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Overview")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{profile['rows']:,}")
    m2.metric("Columns", f"{profile['columns']:,}")
    m3.metric("Missing %", f"{profile['missing_pct']}%")
    m4.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")
    m5.metric("Numeric / Categorical", f"{profile['numeric_cols']} / {profile['categorical_cols']}")

    st.write("Preview")
    st.dataframe(data.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    valid_targets = find_valid_targets(data)

    if not valid_targets:
        st.error("No usable target columns were found.")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Model Setup")

    col1, col2 = st.columns(2)

    with col1:
        target_column = st.selectbox("Select target column", valid_targets)

    with col2:
        detected_task = auto_detect_task(data[target_column])
        task_type = st.radio(
            "Problem type",
            ["classification", "regression"],
            index=0 if detected_task == "classification" else 1,
            horizontal=True,
        )

    available_features = [col for col in data.columns if col != target_column]

    selected_features = st.multiselect(
        "Select feature columns",
        available_features,
        default=available_features,
    )

    tune_model = st.checkbox("Tune supported models with Grid Search", value=False)

    if not selected_features:
        st.warning("Please select at least one feature.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.expander("Inspect target distribution", expanded=False):
        plot_target_distribution(data[target_column], task_type)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run AutoML", type="primary"):
        with st.spinner("Preparing data, comparing models, and training the best pipeline..."):
            df_model = data[selected_features + [target_column]].copy()
            df_model = df_model.drop_duplicates().reset_index(drop=True)

            X = df_model.drop(columns=[target_column])
            y_raw = df_model[target_column]

            valid_idx = y_raw.dropna().index
            X = X.loc[valid_idx].copy()
            y_raw = y_raw.loc[valid_idx].copy()

            if len(X) < 20:
                st.error("Not enough usable rows remain after cleaning. Please upload a larger dataset.")
                return

            label_encoder = None

            if task_type == "regression":
                y = clean_numeric_like_series(y_raw)
                valid_idx = y.dropna().index
                X = X.loc[valid_idx].copy()
                y = y.loc[valid_idx].copy()

                if y.empty:
                    st.error("The selected target could not be used for regression.")
                    return

            else:
                y = y_raw.astype(str).str.strip()
                y = y.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})
                valid_idx = y.dropna().index
                X = X.loc[valid_idx].copy()
                y = y.loc[valid_idx].copy()

                if y.nunique() < 2:
                    st.error("Classification requires at least two target classes.")
                    return

                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            X, preprocessor, numeric_cols, categorical_cols, dropped_summary = build_preprocessor(X)

            if X.empty or len(X.columns) == 0:
                st.error("No valid feature columns remained after cleaning.")
                return

            if len(numeric_cols) == 0 and len(categorical_cols) == 0:
                st.error("No usable features remained after preprocessing.")
                return

            st.subheader("Feature Preparation Summary")
            prep_col1, prep_col2, prep_col3, prep_col4 = st.columns(4)
            prep_col1.metric("Features Used", len(X.columns))
            prep_col2.metric("Numeric Features", len(numeric_cols))
            prep_col3.metric("Categorical Features", len(categorical_cols))
            prep_col4.metric(
                "Dropped Columns",
                len(dropped_summary["high_missing"]) + len(dropped_summary["constant"]) + len(dropped_summary["high_cardinality"]),
            )

            with st.expander("See detected feature types", expanded=False):
                st.write("Numeric columns:", numeric_cols if numeric_cols else "None")
                st.write("Categorical columns:", categorical_cols if categorical_cols else "None")

            with st.expander("See dropped columns", expanded=False):
                st.write("Dropped for high missingness (> 60%):", dropped_summary["high_missing"] or "None")
                st.write("Dropped for being constant:", dropped_summary["constant"] or "None")
                st.write("Dropped for very high-cardinality text / likely IDs:", dropped_summary["high_cardinality"] or "None")

            X_train, X_test, y_train, y_test = safe_train_test_split(X, y, task_type)

            models = get_models(task_type)
            results_df = evaluate_models(X_train, y_train, models, preprocessor, task_type)

            if results_df.empty:
                st.error("All models failed during evaluation.")
                return

            st.subheader("Model Comparison")
            st.dataframe(results_df, use_container_width=True)
            plot_model_results(results_df, task_type)

            best_model_name = results_df.iloc[0]["Model"]
            st.success(f"Best model selected: {best_model_name}")

            if task_type == "classification":
                st.info(
                    "Classification ranking uses balanced accuracy when the target looks imbalanced, which helps prevent misleadingly high accuracy scores."
                )

            best_pipeline = build_pipeline(preprocessor, clone(models[best_model_name]))
            best_params = None

            if tune_model:
                best_pipeline, best_params = run_grid_search(
                    best_model_name,
                    best_pipeline,
                    X_train,
                    y_train,
                    task_type,
                )

            best_pipeline.fit(X_train, y_train)
            y_pred = best_pipeline.predict(X_test)

            st.subheader("Holdout Test Performance")

            if task_type == "classification":
                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Accuracy", f"{acc:.3f}")
                mc2.metric("Balanced Accuracy", f"{bal_acc:.3f}")
                mc3.metric("Precision", f"{prec:.3f}")
                mc4.metric("Recall", f"{rec:.3f}")
                mc5.metric("F1 Score", f"{f1:.3f}")

                st.write("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                st.dataframe(pd.DataFrame(cm), use_container_width=True)

                if label_encoder is not None:
                    with st.expander("Encoded class labels", expanded=False):
                        st.write(
                            pd.DataFrame(
                                {
                                    "Encoded Value": range(len(label_encoder.classes_)),
                                    "Original Label": label_encoder.classes_,
                                }
                            )
                        )

            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                mr1, mr2, mr3 = st.columns(3)
                mr1.metric("RMSE", f"{rmse:.3f}")
                mr2.metric("MAE", f"{mae:.3f}")
                mr3.metric("R²", f"{r2:.3f}")

            artifact = {
                "pipeline": best_pipeline,
                "task_type": task_type,
                "target_column": target_column,
                "selected_features": selected_features,
                "numeric_features_after_cleaning": numeric_cols,
                "categorical_features_after_cleaning": categorical_cols,
                "dropped_columns_summary": dropped_summary,
                "best_model_name": best_model_name,
                "best_model_params": best_params,
            }

            if task_type == "classification" and label_encoder is not None:
                artifact["label_encoder"] = label_encoder

            buffer = io.BytesIO()
            joblib.dump(artifact, buffer)
            buffer.seek(0)

            st.download_button(
                "Download Trained Pipeline",
                data=buffer,
                file_name="best_automl_pipeline.pkl",
                mime="application/octet-stream",
            )

            st.success("Done. Your best model has been trained and packaged for download.")


# ============================================================
# APP ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()

# streamlit run automated_ml_tool.py
