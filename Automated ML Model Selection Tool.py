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
import inspect
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
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
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
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

# Silence noisy warnings for cleaner public app experience
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

# ------------------------------------------------------------
# PROFESSIONAL UI STYLING
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Main app background */
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(49, 130, 206, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.12), transparent 24%),
                linear-gradient(180deg, #06111f 0%, #0b1728 42%, #0f1f35 100%);
            color: #E6EEF8;
        }

        /* Main container spacing */
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.25rem;
            max-width: 1300px;
        }

        /* Typography */
        h1, h2, h3, h4 {
            color: #F8FBFF;
            letter-spacing: 0.2px;
        }

        p, label, span, div {
            color: #D7E3F3;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(7,19,34,0.98), rgba(13,29,50,0.98));
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #E7F0FA !important;
        }

        /* Hero card */
        .hero-card {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(36, 78, 135, 0.55), rgba(10, 18, 40, 0.80)),
                linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(6, 182, 212, 0.12));
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 24px;
            padding: 1.6rem 1.6rem 1.35rem 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 20px 45px rgba(0,0,0,0.28);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            top: -80px;
            right: -80px;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(56,189,248,0.20), transparent 70%);
            pointer-events: none;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            color: #FFFFFF;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #DCE8F8;
            margin-bottom: 0.55rem;
        }

        .hero-note {
            font-size: 0.92rem;
            color: #B9CBE0;
            margin-bottom: 0;
        }

        /* General content cards */
        .section-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.025));
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 20px;
            padding: 1.15rem 1.15rem 0.95rem 1.15rem;
            margin-bottom: 1.15rem;
            box-shadow: 0 14px 35px rgba(0,0,0,0.18);
            backdrop-filter: blur(8px);
        }

        .section-title {
            font-size: 1.12rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: #F7FBFF;
        }

        .section-subtitle {
            font-size: 0.94rem;
            color: #B8C7DA;
            margin-bottom: 0.9rem;
        }

        /* Feature highlight strip */
        .mini-banner {
            background: linear-gradient(90deg, rgba(59,130,246,0.12), rgba(34,197,94,0.07));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
        }

        .mini-banner strong {
            color: #F8FBFF;
        }

        /* Upload card */
        .upload-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
            border: 1px dashed rgba(148, 163, 184, 0.35);
            border-radius: 18px;
            padding: 0.6rem 0.8rem 0.3rem 0.8rem;
            margin-bottom: 1rem;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.08);
            padding: 0.95rem;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }

        div[data-testid="stMetricLabel"] {
            color: #AFC4DB !important;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-weight: 800;
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            overflow: hidden;
            background: rgba(255,255,255,0.02);
        }

        /* Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea,
        input {
            border-radius: 12px !important;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            font-weight: 700;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(135deg, #2563EB, #0EA5E9);
            color: white;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.28);
        }

        .stButton > button:hover {
            border-color: rgba(255,255,255,0.14);
            box-shadow: 0 12px 26px rgba(14, 165, 233, 0.32);
        }

        .stDownloadButton > button {
            width: 100%;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            font-weight: 700;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(135deg, #059669, #10B981);
            color: white;
            box-shadow: 0 10px 22px rgba(16, 185, 129, 0.25);
        }

        .stDownloadButton > button:hover {
            border-color: rgba(255,255,255,0.14);
        }

        /* Expanders */
        details {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 0.25rem 0.55rem;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            border-radius: 12px 12px 0 0 !important;
        }

        /* Footer */
        .footer-card {
            margin-top: 1.25rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            padding: 0.95rem 1.1rem;
            color: #BFD0E3;
            font-size: 0.93rem;
        }

        /* Small muted text */
        .tiny-note {
            color: #AFC1D7;
            font-size: 0.92rem;
        }

        /* Divider line look */
        hr {
            border: none;
            height: 1px;
            background: rgba(255,255,255,0.08);
            margin: 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS: COMPATIBILITY / SAFETY
# ============================================================
def make_onehot_encoder():
    """
    Create a version-compatible OneHotEncoder.

    Why this is needed:
    - Newer sklearn uses: sparse_output=False
    - Older sklearn uses: sparse=False

    This avoids a version crash in public deployments.
    """
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def safe_float(value, default=np.nan):
    """
    Convert a value to float safely.
    """
    try:
        return float(value)
    except Exception:
        return default


def summarize_exception(e: Exception) -> str:
    """
    Create a short readable error string for the UI.
    """
    msg = str(e).strip()
    if not msg:
        msg = e.__class__.__name__
    return f"{e.__class__.__name__}: {msg}"


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
    Robust Excel reader with engine fallbacks.
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
        "Excel file could not be read. Make sure the file is valid and the required engine is installed. "
        f"Last error: {last_error}"
    )


def try_read_json(uploaded_file) -> pd.DataFrame:
    """
    Robust JSON reader.

    Handles:
    - normal JSON
    - records-style JSON
    - line-delimited JSON
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
    Decide whether the first row likely contains headers.
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

        # Convert strongly numeric-looking object columns into numeric columns
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

    Safer than the original logic because it avoids treating common
    numeric-coded class labels (0/1/2, 1/2/3, etc.) as regression.
    """
    y_non_null = y_series.dropna()

    if y_non_null.empty:
        return "classification"

    # If target is truly non-numeric, it is classification
    if not pd.api.types.is_numeric_dtype(y_non_null):
        numeric_candidate = clean_numeric_like_series(y_non_null)
        conversion_ratio = numeric_candidate.notna().mean()

        if conversion_ratio < 0.90:
            return "classification"

        y_non_null = numeric_candidate.dropna()

    # At this point target is numeric or numeric-like
    unique_count = y_non_null.nunique()
    n_rows = len(y_non_null)

    # Strongly prefer classification for very low-cardinality numeric targets
    if unique_count <= 10:
        return "classification"

    # Integer targets with a small ratio of unique values are often classes
    is_integer_like = np.allclose(y_non_null, np.round(y_non_null), equal_nan=True)
    unique_ratio = unique_count / max(n_rows, 1)

    if is_integer_like and unique_ratio <= 0.05:
        return "classification"

    return "regression"


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

    for col in list(X.columns):
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
    Standardize mixed-type columns without destroying numeric information.
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
            ("onehot", make_onehot_encoder()),
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
# HELPER FUNCTIONS: TARGET VALIDATION
# ============================================================
def validate_classification_target(y: pd.Series) -> Tuple[bool, str]:
    """
    Validate classification target before modeling.
    """
    y = pd.Series(y).dropna()

    if y.empty:
        return False, "The classification target is empty after cleaning."

    if y.nunique() < 2:
        return False, "Classification requires at least two target classes."

    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        return True, "One or more classes have fewer than 2 samples. Some models may be unstable."

    return True, ""


def validate_regression_target(y: pd.Series) -> Tuple[bool, str]:
    """
    Validate regression target before modeling.
    """
    y = pd.Series(y).dropna()

    if y.empty:
        return False, "The regression target is empty after cleaning."

    if y.nunique() < 5:
        return True, "This numeric target has very few unique values. Double-check that this is truly regression."

    return True, ""


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
    Use balanced accuracy on imbalanced targets and F1 weighted otherwise.
    """
    return "balanced_accuracy" if is_imbalanced_classification(y_train) else "f1_weighted"


def classify_reliability_level(score: float) -> str:
    """
    Convert a numeric reliability score into a plain-English label.

    This is intentionally conservative because the app is public-facing.
    A model should not be presented as reliable simply because it has
    the highest score among weak alternatives.
    """
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Moderate"
    if score >= 40:
        return "Caution"
    return "Weak"


def compute_reliability_score(
    task_type: str,
    best_holdout_metrics: Dict[str, float],
    baseline_holdout_metrics: Dict[str, float],
    cv_row: pd.Series,
    n_rows: int,
    n_features: int,
    failed_model_count: int,
) -> Tuple[int, List[str]]:
    """
    Score the reliability of the selected model from 0 to 100.

    This is not a scientific guarantee. It is a practical public-facing
    safety layer that checks whether the selected model:
    - beats a simple baseline,
    - has stable cross-validation results,
    - has enough data,
    - avoids obviously weak holdout performance,
    - did not win only because many algorithms failed.
    """
    score = 100
    warnings_list = []

    # Penalize very small datasets because holdout and CV estimates become unstable.
    if n_rows < 50:
        score -= 25
        warnings_list.append("Very small dataset. Results may change significantly with more data.")
    elif n_rows < 150:
        score -= 12
        warnings_list.append("Small dataset. Treat the result as an early baseline, not a final production model.")

    # Too many features compared with rows can cause overfitting.
    if n_features > 0 and n_rows / max(n_features, 1) < 5:
        score -= 12
        warnings_list.append("There are many features relative to the number of rows. Overfitting risk is higher.")

    if failed_model_count > 0:
        score -= min(15, failed_model_count * 3)
        warnings_list.append(f"{failed_model_count} model(s) failed during evaluation and were excluded.")

    # Cross-validation variability check.
    cv_std = safe_float(cv_row.get("Primary CV Std", np.nan))
    cv_mean = safe_float(cv_row.get("Primary CV Score", np.nan))

    if not np.isnan(cv_std):
        if task_type == "classification":
            if cv_std > 0.10:
                score -= 12
                warnings_list.append("Cross-validation scores vary a lot across folds. Model stability is questionable.")
        else:
            # For RMSE, use coefficient of variation when possible.
            if cv_mean > 0 and (cv_std / cv_mean) > 0.25:
                score -= 12
                warnings_list.append("Regression error varies a lot across folds. Model stability is questionable.")

    if task_type == "classification":
        f1_macro_value = best_holdout_metrics.get("f1_macro", np.nan)
        balanced_accuracy_value = best_holdout_metrics.get("balanced_accuracy", np.nan)

        baseline_bal_acc = baseline_holdout_metrics.get("balanced_accuracy", np.nan)
        improvement = balanced_accuracy_value - baseline_bal_acc

        if not np.isnan(improvement) and improvement < 0.03:
            score -= 25
            warnings_list.append("The selected classifier barely improves over the dummy baseline.")

        if not np.isnan(f1_macro_value) and f1_macro_value < 0.50:
            score -= 15
            warnings_list.append("Macro F1 is low, meaning at least some classes may be performing poorly.")

        if not np.isnan(balanced_accuracy_value) and balanced_accuracy_value < 0.60:
            score -= 15
            warnings_list.append("Balanced accuracy is low. Accuracy alone may be misleading for this dataset.")

    else:
        rmse = best_holdout_metrics.get("rmse", np.nan)
        baseline_rmse = baseline_holdout_metrics.get("rmse", np.nan)
        r2_value = best_holdout_metrics.get("r2", np.nan)

        if not np.isnan(rmse) and not np.isnan(baseline_rmse):
            if rmse >= baseline_rmse:
                score -= 35
                warnings_list.append("The selected regressor does not beat the mean baseline on holdout RMSE.")
            else:
                improvement_pct = ((baseline_rmse - rmse) / baseline_rmse) * 100 if baseline_rmse != 0 else 0
                if improvement_pct < 5:
                    score -= 18
                    warnings_list.append("The selected regressor improves only slightly over the mean baseline.")

        if not np.isnan(r2_value) and r2_value < 0:
            score -= 20
            warnings_list.append("R² is negative, meaning the model is worse than a simple mean prediction on this test split.")

    score = int(max(0, min(100, score)))
    return score, warnings_list


def build_reliability_report(
    task_type: str,
    best_model_name: str,
    best_holdout_metrics: Dict[str, float],
    baseline_holdout_metrics: Dict[str, float],
    cv_row: pd.Series,
    n_rows: int,
    n_features: int,
    failed_model_count: int,
) -> Dict[str, object]:
    """
    Build a structured reliability report for display and export.

    This gives users a clear interpretation instead of only showing raw scores.
    """
    score, warnings_list = compute_reliability_score(
        task_type=task_type,
        best_holdout_metrics=best_holdout_metrics,
        baseline_holdout_metrics=baseline_holdout_metrics,
        cv_row=cv_row,
        n_rows=n_rows,
        n_features=n_features,
        failed_model_count=failed_model_count,
    )

    return {
        "best_model_name": best_model_name,
        "task_type": task_type,
        "reliability_score": score,
        "reliability_level": classify_reliability_level(score),
        "warnings": warnings_list,
        "cv_primary_score": safe_float(cv_row.get("Primary CV Score", np.nan)),
        "cv_primary_std": safe_float(cv_row.get("Primary CV Std", np.nan)),
        "holdout_metrics": best_holdout_metrics,
        "baseline_metrics": baseline_holdout_metrics,
    }


def display_reliability_report(report: Dict[str, object]) -> None:
    """
    Display the model reliability report in Streamlit.
    """
    st.markdown("#### Reliability Verdict")

    col_a, col_b = st.columns(2)
    col_a.metric("Reliability Score", f"{report['reliability_score']}/100")
    col_b.metric("Reliability Level", report["reliability_level"])

    warnings_list = report.get("warnings", [])

    if report["reliability_level"] in {"Strong", "Moderate"} and not warnings_list:
        st.success("The selected model passed the main reliability checks for this dataset.")
    elif report["reliability_level"] == "Moderate":
        st.info("The selected model is usable as a baseline, but users should still review the warnings below.")
    elif report["reliability_level"] == "Caution":
        st.warning("Use caution. The model may be useful, but the results should not be presented as highly reliable.")
    else:
        st.error("Weak reliability. This model should not be presented to users as a dependable prediction system yet.")

    if warnings_list:
        st.markdown("##### Reliability warnings")
        for item in warnings_list:
            st.warning(item)



def get_models(task_type: str) -> Dict[str, object]:
    """
    Return candidate models based on task type.
    """
    if task_type == "classification":
        models = {
            "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(
                max_iter=10000,
                class_weight="balanced",
                solver="lbfgs",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
            "SVM": SVC(probability=True, class_weight="balanced"),
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
            "Dummy Baseline": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
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


def filter_models_for_data(models: Dict[str, object], X_train, y_train, task_type: str) -> Dict[str, object]:
    """
    Remove models that are likely to fail on the current dataset.

    This prevents public-facing crashes on tiny class counts or
    very small sample sizes.
    """
    filtered = {}

    n_samples = len(X_train)

    if task_type == "classification":
        y_series = pd.Series(y_train)
        min_class_count = y_series.value_counts().min()
        n_classes = y_series.nunique()

        for name, model in models.items():
            # KNN can fail if neighbors > class-fold sample sizes
            if name == "KNN" and n_samples < 10:
                continue

            # Very tiny classes can make some models unstable
            if name in {"SVM", "KNN"} and min_class_count < 2:
                continue

            # Some boosting libraries can be touchy in edge multiclass cases
            if name in {"XGBoost", "LightGBM"} and n_classes < 2:
                continue

            filtered[name] = model

    else:
        for name, model in models.items():
            if name == "KNN" and n_samples < 10:
                continue
            filtered[name] = model

    return filtered


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
    If tuning fails, return the original pipeline.
    """
    param_grid = get_param_grid(model_name, task_type)

    if not param_grid:
        st.info(f"No tuning grid defined for {model_name}. Using default settings.")
        return pipeline, None

    try:
        cv_strategy = get_cv_strategy(y_train, task_type, preferred_folds=4)

        if task_type == "classification":
            scoring = get_classification_primary_metric(y_train)
        else:
            scoring = "neg_root_mean_squared_error"

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        grid.fit(X_train, y_train)

        st.success(f"Best parameters for {model_name}: {grid.best_params_}")
        return grid.best_estimator_, grid.best_params_

    except Exception as e:
        st.warning(f"Grid search skipped for {model_name}: {summarize_exception(e)}")
        return pipeline, None


def evaluate_models(X_train, y_train, models: Dict[str, object], preprocessor, task_type: str):
    """
    Evaluate candidate models using cross-validation.

    Important improvement:
    - We do NOT silently hide model failures anymore.
    - Failed models are reported back to the UI.
    """
    rows = []
    failures = []

    cv_strategy = get_cv_strategy(y_train, task_type, preferred_folds=5)

    for name, model in models.items():
        try:
            pipeline = build_pipeline(preprocessor, clone(model))

            if task_type == "classification":
                primary_metric = get_classification_primary_metric(y_train)
                scoring = {
                    "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy",
                    "precision_weighted": "precision_weighted",
                    "recall_weighted": "recall_weighted",
                    "f1_weighted": "f1_weighted",
                    "f1_macro": "f1_macro",
                }

                cv_results = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_strategy,
                    scoring=scoring,
                    n_jobs=-1,
                    error_score="raise",
                )

                rows.append(
                    {
                        "Model": name,
                        "Primary CV Score": float(np.mean(cv_results[f"test_{primary_metric}"])),
                        "Primary CV Std": float(np.std(cv_results[f"test_{primary_metric}"])),
                        "Score Type": "Balanced Accuracy" if primary_metric == "balanced_accuracy" else "F1 Weighted",
                        "Accuracy": float(np.mean(cv_results["test_accuracy"])),
                        "Accuracy Std": float(np.std(cv_results["test_accuracy"])),
                        "Balanced Accuracy": float(np.mean(cv_results["test_balanced_accuracy"])),
                        "Balanced Accuracy Std": float(np.std(cv_results["test_balanced_accuracy"])),
                        "Precision Weighted": float(np.mean(cv_results["test_precision_weighted"])),
                        "Recall Weighted": float(np.mean(cv_results["test_recall_weighted"])),
                        "F1 Weighted": float(np.mean(cv_results["test_f1_weighted"])),
                        "F1 Macro": float(np.mean(cv_results["test_f1_macro"])),
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
                    n_jobs=-1,
                    error_score="raise",
                )

                rmse_cv = float(-np.mean(cv_results["test_rmse"]))
                mae_cv = float(-np.mean(cv_results["test_mae"]))
                r2_cv = float(np.mean(cv_results["test_r2"]))

                rows.append(
                    {
                        "Model": name,
                        "Primary CV Score": rmse_cv,
                        "Primary CV Std": float(np.std(-cv_results["test_rmse"])),
                        "Score Type": "RMSE",
                        "RMSE": rmse_cv,
                        "RMSE Std": float(np.std(-cv_results["test_rmse"])),
                        "MAE": mae_cv,
                        "MAE Std": float(np.std(-cv_results["test_mae"])),
                        "R²": r2_cv,
                        "R² Std": float(np.std(cv_results["test_r2"])),
                    }
                )

        except Exception as e:
            failures.append(
                {
                    "Model": name,
                    "Status": "Failed",
                    "Reason": summarize_exception(e),
                }
            )

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        ascending = task_type == "regression"
        results_df = results_df.sort_values("Primary CV Score", ascending=ascending).reset_index(drop=True)

    failures_df = pd.DataFrame(failures)
    return results_df, failures_df


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
    Plot a simple target distribution.
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


def plot_actual_vs_predicted(y_test, y_pred) -> None:
    """
    Plot actual vs predicted values for regression tasks.
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.7)

    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
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


def get_prediction_scores(model, X_test):
    """
    Safely get probability or decision scores for ROC AUC when available.
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]

        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            if isinstance(scores, np.ndarray):
                if scores.ndim == 1:
                    return scores
                if scores.ndim == 2 and scores.shape[1] >= 2:
                    return scores[:, 1]
    except Exception:
        pass

    return None


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # --------------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">AutoML Model Selection Tool</div>
            <div class="hero-subtitle">
                Upload a dataset, choose a target, compare reliable baseline models,
                and export your best trained machine learning pipeline.
            </div>
            <div class="hero-note">
                Built for real-world data with safer preprocessing, task-aware metrics, and a cleaner public-facing experience.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mini-banner">
            <strong>What this tool helps you do:</strong>
            clean messy data, detect feature types, compare strong baseline models, evaluate them properly,
            and download the best trained end-to-end pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------
    st.sidebar.markdown("## Workflow")
    st.sidebar.markdown(
        """
        **1. Upload data**  
        **2. Choose target and features**  
        **3. Compare models**  
        **4. Review results**  
        **5. Download trained pipeline**
        """
    )

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

    with st.sidebar.expander("Professional note", expanded=False):
        st.write(
            """
            This interface focuses on model comparison and strong baseline selection.
            Strong metrics still depend on data quality, feature relevance, and target choice.
            """
        )

    # --------------------------------------------------------
    # UPLOAD SECTION
    # --------------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Start by uploading a structured dataset for automated model analysis.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "xlsm", "xlsb", "json"],
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded_file:
        st.info("Upload a dataset to begin.")
        st.markdown(
            """
            <div class="footer-card">
                Ready for use once a dataset is uploaded. The interface will guide you through setup, evaluation, and export.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    data = load_and_clean_data(uploaded_file)

    if data is None or data.empty:
        st.error("The uploaded file could not be processed.")
        return

    profile = get_dataset_profile(data)

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A quick profile of the uploaded data before modeling begins.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{profile['rows']:,}")
    m2.metric("Columns", f"{profile['columns']:,}")
    m3.metric("Missing %", f"{profile['missing_pct']}%")
    m4.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")
    m5.metric("Numeric / Categorical", f"{profile['numeric_cols']} / {profile['categorical_cols']}")

    st.markdown("#### Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    valid_targets = find_valid_targets(data)

    if not valid_targets:
        st.error("No usable target columns were found.")
        return

    # --------------------------------------------------------
    # MODEL SETUP
    # --------------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Setup</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Choose the target, confirm the task type, and select the features for training.</div>',
        unsafe_allow_html=True,
    )

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

    # --------------------------------------------------------
    # RUN SECTION
    # --------------------------------------------------------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Run Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Launch preprocessing, model comparison, training, evaluation, and pipeline export.</div>',
        unsafe_allow_html=True,
    )
    run_automl = st.button("Run AutoML", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_automl:
        with st.spinner("Preparing data, comparing models, and training the best pipeline..."):
            df_model = data[selected_features + [target_column]].copy()
            df_model = df_model.drop_duplicates().reset_index(drop=True)

            X = df_model.drop(columns=[target_column])
            y_raw = df_model[target_column]

            # Drop rows with missing target first
            valid_idx = y_raw.dropna().index
            X = X.loc[valid_idx].copy()
            y_raw = y_raw.loc[valid_idx].copy()

            if len(X) < 20:
                st.error("Not enough usable rows remain after cleaning. Please upload a larger dataset.")
                return

            label_encoder = None

            # -----------------------------
            # Prepare target based on task
            # -----------------------------
            if task_type == "regression":
                y = clean_numeric_like_series(y_raw)
                valid_idx = y.dropna().index
                X = X.loc[valid_idx].copy()
                y = y.loc[valid_idx].copy()

                ok, message = validate_regression_target(y)
                if not ok:
                    st.error(message)
                    return
                if message:
                    st.warning(message)

            else:
                y = y_raw.astype(str).str.strip()
                y = y.replace(
                    {
                        "": np.nan,
                        "nan": np.nan,
                        "NaN": np.nan,
                        "None": np.nan,
                        "none": np.nan,
                        "null": np.nan,
                        "NULL": np.nan,
                    }
                )

                valid_idx = y.dropna().index
                X = X.loc[valid_idx].copy()
                y = y.loc[valid_idx].copy()

                ok, message = validate_classification_target(y)
                if not ok:
                    st.error(message)
                    return
                if message:
                    st.warning(message)

                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            # -----------------------------
            # Build preprocessing
            # -----------------------------
            X, preprocessor, numeric_cols, categorical_cols, dropped_summary = build_preprocessor(X)

            if X.empty or len(X.columns) == 0:
                st.error("No valid feature columns remained after cleaning.")
                return

            if len(numeric_cols) == 0 and len(categorical_cols) == 0:
                st.error("No usable features remained after preprocessing.")
                return

            # ------------------------------------------------
            # FEATURE PREPARATION SUMMARY
            # ------------------------------------------------
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Feature Preparation Summary</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">A breakdown of usable features after cleaning and preprocessing.</div>',
                unsafe_allow_html=True,
            )

            prep_col1, prep_col2, prep_col3, prep_col4 = st.columns(4)
            prep_col1.metric("Features Used", len(X.columns))
            prep_col2.metric("Numeric Features", len(numeric_cols))
            prep_col3.metric("Categorical Features", len(categorical_cols))
            prep_col4.metric(
                "Dropped Columns",
                len(dropped_summary["high_missing"]) +
                len(dropped_summary["constant"]) +
                len(dropped_summary["high_cardinality"]),
            )

            with st.expander("See detected feature types", expanded=False):
                st.write("Numeric columns:", numeric_cols if numeric_cols else "None")
                st.write("Categorical columns:", categorical_cols if categorical_cols else "None")

            with st.expander("See dropped columns", expanded=False):
                st.write("Dropped for high missingness (> 60%):", dropped_summary["high_missing"] or "None")
                st.write("Dropped for being constant:", dropped_summary["constant"] or "None")
                st.write("Dropped for very high-cardinality text / likely IDs:", dropped_summary["high_cardinality"] or "None")

            st.markdown("</div>", unsafe_allow_html=True)

            # -----------------------------
            # Split data
            # -----------------------------
            X_train, X_test, y_train, y_test = safe_train_test_split(X, y, task_type)

            # -----------------------------
            # Select and filter models
            # -----------------------------
            models = get_models(task_type)
            models = filter_models_for_data(models, X_train, y_train, task_type)

            if not models:
                st.error("No models are suitable for this dataset after safety checks.")
                return

            # -----------------------------
            # Cross-validated comparison
            # -----------------------------
            results_df, failures_df = evaluate_models(X_train, y_train, models, preprocessor, task_type)

            if results_df.empty:
                st.error("All models failed during evaluation.")
                return

            # ------------------------------------------------
            # MODEL COMPARISON
            # ------------------------------------------------
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Cross-validated model performance using task-appropriate ranking logic.</div>',
                unsafe_allow_html=True,
            )

            if not failures_df.empty:
                with st.expander("Models that failed during evaluation", expanded=False):
                    st.dataframe(failures_df, use_container_width=True)

            st.dataframe(results_df, use_container_width=True)
            plot_model_results(results_df, task_type)

            best_model_name = results_df.iloc[0]["Model"]
            best_cv_row = results_df.iloc[0].copy()

            st.success(f"Best model selected: {best_model_name}")

            if best_model_name == "Dummy Baseline":
                st.warning(
                    "The dummy baseline was selected as the best model. This is an honest signal that the current features may not contain enough predictive power."
                )

            if task_type == "classification":
                st.info(
                    "Classification ranking uses Balanced Accuracy for imbalanced targets and F1 Weighted otherwise, helping reduce misleadingly optimistic results."
                )
            else:
                st.info(
                    "Regression ranking uses the lowest cross-validated RMSE, which is safer than relying on R² alone."
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # -----------------------------
            # Train best model
            # -----------------------------
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

            try:
                best_pipeline.fit(X_train, y_train)
            except Exception as e:
                st.error(f"Best model training failed: {summarize_exception(e)}")
                return

            try:
                y_pred = best_pipeline.predict(X_test)
            except Exception as e:
                st.error(f"Prediction failed: {summarize_exception(e)}")
                return

            # ------------------------------------------------
            # HOLDOUT TEST PERFORMANCE
            # ------------------------------------------------
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Holdout Test Performance</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Final evaluation of the selected model on unseen test data.</div>',
                unsafe_allow_html=True,
            )

            # -----------------------------
            # Classification metrics
            # -----------------------------
            if task_type == "classification":
                acc = accuracy_score(y_test, y_pred)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

                # Baseline comparison for classification.
                # This protects users from being impressed by raw accuracy when a
                # simple majority-class classifier performs almost as well.
                baseline_model = DummyClassifier(strategy="most_frequent")
                baseline_pipeline = build_pipeline(preprocessor, baseline_model)
                baseline_pipeline.fit(X_train, y_train)
                baseline_pred = baseline_pipeline.predict(X_test)

                baseline_acc = accuracy_score(y_test, baseline_pred)
                baseline_bal_acc = balanced_accuracy_score(y_test, baseline_pred)
                baseline_f1_w = f1_score(y_test, baseline_pred, average="weighted", zero_division=0)
                baseline_f1_m = f1_score(y_test, baseline_pred, average="macro", zero_division=0)

                best_holdout_metrics = {
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "precision_weighted": prec,
                    "recall_weighted": rec,
                    "f1_weighted": f1_w,
                    "f1_macro": f1_m,
                }

                baseline_holdout_metrics = {
                    "accuracy": baseline_acc,
                    "balanced_accuracy": baseline_bal_acc,
                    "f1_weighted": baseline_f1_w,
                    "f1_macro": baseline_f1_m,
                }

                mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                mc1.metric("Accuracy", f"{acc:.3f}")
                mc2.metric("Balanced Accuracy", f"{bal_acc:.3f}")
                mc3.metric("Precision", f"{prec:.3f}")
                mc4.metric("Recall", f"{rec:.3f}")
                mc5.metric("F1 Weighted", f"{f1_w:.3f}")
                mc6.metric("F1 Macro", f"{f1_m:.3f}")

                with st.expander("Baseline comparison", expanded=False):
                    st.write(
                        pd.DataFrame(
                            {
                                "Metric": ["Accuracy", "Balanced Accuracy", "F1 Weighted", "F1 Macro"],
                                "Best Model": [acc, bal_acc, f1_w, f1_m],
                                "Dummy Baseline": [baseline_acc, baseline_bal_acc, baseline_f1_w, baseline_f1_m],
                            }
                        )
                    )

                reliability_report = build_reliability_report(
                    task_type=task_type,
                    best_model_name=best_model_name,
                    best_holdout_metrics=best_holdout_metrics,
                    baseline_holdout_metrics=baseline_holdout_metrics,
                    cv_row=best_cv_row,
                    n_rows=len(X),
                    n_features=len(X.columns),
                    failed_model_count=len(failures_df),
                )
                display_reliability_report(reliability_report)

                # ROC AUC only makes sense safely in binary classification
                if len(np.unique(y_test)) == 2:
                    score_values = get_prediction_scores(best_pipeline, X_test)
                    if score_values is not None:
                        try:
                            auc = roc_auc_score(y_test, score_values)
                            st.metric("ROC AUC", f"{auc:.3f}")
                        except Exception:
                            pass

                st.markdown("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm)
                st.dataframe(cm_df, use_container_width=True)

                st.markdown("#### Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)

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

            # -----------------------------
            # Regression metrics
            # -----------------------------
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Baseline comparison
                baseline_model = DummyRegressor(strategy="mean")
                baseline_pipeline = build_pipeline(preprocessor, baseline_model)
                baseline_pipeline.fit(X_train, y_train)
                baseline_pred = baseline_pipeline.predict(X_test)
                baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
                baseline_mae = mean_absolute_error(y_test, baseline_pred)
                baseline_r2 = r2_score(y_test, baseline_pred)

                mr1, mr2, mr3, mr4 = st.columns(4)
                mr1.metric("RMSE", f"{rmse:.3f}")
                mr2.metric("MAE", f"{mae:.3f}")
                mr3.metric("R²", f"{r2:.3f}")
                mr4.metric("Baseline RMSE", f"{baseline_rmse:.3f}")

                with st.expander("Baseline comparison", expanded=False):
                    st.write(
                        pd.DataFrame(
                            {
                                "Metric": ["RMSE", "MAE", "R²"],
                                "Best Model": [rmse, mae, r2],
                                "Baseline": [baseline_rmse, baseline_mae, baseline_r2],
                            }
                        )
                    )

                best_holdout_metrics = {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }

                baseline_holdout_metrics = {
                    "rmse": baseline_rmse,
                    "mae": baseline_mae,
                    "r2": baseline_r2,
                }

                reliability_report = build_reliability_report(
                    task_type=task_type,
                    best_model_name=best_model_name,
                    best_holdout_metrics=best_holdout_metrics,
                    baseline_holdout_metrics=baseline_holdout_metrics,
                    cv_row=best_cv_row,
                    n_rows=len(X),
                    n_features=len(X.columns),
                    failed_model_count=len(failures_df),
                )
                display_reliability_report(reliability_report)

                if rmse >= baseline_rmse:
                    st.warning(
                        "The selected regression model did not outperform the mean baseline on the holdout set. This suggests the dataset may need better features, more cleaning, or a different target."
                    )

                plot_actual_vs_predicted(y_test, y_pred)

            st.markdown("</div>", unsafe_allow_html=True)

            # -----------------------------
            # Save artifact
            # -----------------------------
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
                "reliability_report": reliability_report,
            }

            if task_type == "classification" and label_encoder is not None:
                artifact["label_encoder"] = label_encoder

            buffer = io.BytesIO()
            joblib.dump(artifact, buffer)
            buffer.seek(0)

            # ------------------------------------------------
            # DOWNLOAD SECTION
            # ------------------------------------------------
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Export Best Pipeline</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Download the fitted preprocessing-and-model pipeline for reuse on future data.</div>',
                unsafe_allow_html=True,
            )

            st.download_button(
                "Download Trained Pipeline",
                data=buffer,
                file_name="best_automl_pipeline.pkl",
                mime="application/octet-stream",
            )

            st.success("Done. Your best model has been trained and packaged for download.")
            st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="footer-card">
            <strong>AutoML Model Selection Tool</strong><br>
            Professional UI layer with safer preprocessing, task-aware evaluation, and downloadable trained pipelines.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APP ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()

# streamlit run automated_ml_tool.py
