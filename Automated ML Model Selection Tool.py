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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more polished, launch-ready interface
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #07111f 0%, #0b1728 45%, #0f2036 100%);
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

        .metric-label {
            font-size: 0.95rem;
            color: #c8d4e3;
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
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS: DATA LOADING / CLEANING
# ============================================================
def make_column_names_unique(columns: List[str]) -> List[str]:
    """
    Clean and make column names unique.

    Why this matters:
    Real datasets often contain duplicate headers, blank headers,
    or headers with trailing spaces/newlines.

    Example:
    ["Age", "Age", " Salary "] becomes ["Age", "Age_1", "Salary"]
    """
    counts = {}
    new_columns = []

    for col in columns:
        # Convert any header object to string and clean it
        clean_col = str(col).strip().replace("\n", " ")

        # Replace empty-like headers with a default placeholder
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
    Assign a simple score to a row to determine whether it looks like a header.

    Higher score = more likely to be a real header row.

    Signals that help a row look like a header:
    - values are mostly strings
    - not too many missing or unnamed cells
    - values are reasonably unique
    """
    row_str = row.fillna("").astype(str).str.strip()

    if len(row_str) == 0:
        return -999

    bad_tokens = {"", "nan", "null", "none", "unnamed"}
    bad_ratio = row_str.str.lower().isin(bad_tokens).mean()
    unique_ratio = row_str.nunique(dropna=False) / max(len(row_str), 1)
    alpha_ratio = row_str.apply(lambda x: any(ch.isalpha() for ch in x)).mean()

    # Weighted heuristic score
    score = (alpha_ratio * 0.5) + (unique_ratio * 0.35) - (bad_ratio * 0.9)
    return score


def detect_header_row(raw_data: pd.DataFrame, max_rows_to_check: int = 5) -> int:
    """
    Detect which row most likely contains the headers.

    We inspect the first few rows and choose the one with the best score.
    This makes the app more forgiving when files contain title rows or exports
    with metadata above the actual header row.
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
    """
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, engine="openpyxl", header=None)


def try_read_json(uploaded_file) -> pd.DataFrame:
    """
    Robust JSON reader.
    """
    uploaded_file.seek(0)
    data = pd.read_json(uploaded_file)

    # If the JSON loads into a dictionary-like structure,
    # normalize nested records when possible.
    if isinstance(data, dict):
        return pd.json_normalize(data)

    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame(data)

    return data


def load_and_clean_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load the uploaded dataset and apply first-pass cleaning.

    Supported formats:
    - CSV
    - XLSX
    - JSON

    Cleaning steps:
    - detect likely header row
    - create unique cleaned column names
    - remove fully empty rows and columns
    - reset index
    """
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            raw_data = try_read_csv(uploaded_file)
        elif filename.endswith(".xlsx"):
            raw_data = try_read_excel(uploaded_file)
        elif filename.endswith(".json"):
            raw_data = try_read_json(uploaded_file)
            # If already structured correctly with real headers,
            # lightly clean and return early.
            if isinstance(raw_data, pd.DataFrame) and raw_data.columns.dtype != int:
                raw_data.columns = make_column_names_unique(list(raw_data.columns))
                raw_data = raw_data.dropna(axis=1, how="all").dropna(axis=0, how="all")
                return raw_data.reset_index(drop=True)
        else:
            raise ValueError("Unsupported file format. Please upload CSV, XLSX, or JSON.")

        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        if raw_data.empty:
            raise ValueError("The uploaded file appears to be empty.")

        header_row = detect_header_row(raw_data)
        data = raw_data.iloc[header_row + 1:].copy()

        # Use the chosen header row as column names
        detected_columns = raw_data.iloc[header_row].tolist()
        data.columns = make_column_names_unique(detected_columns)

        # Remove fully empty rows and columns
        data = data.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

        # Light first-pass conversion for numeric-looking columns
        # This stays conservative and avoids expensive operations.
        for col in data.columns:
            if data[col].dtype == "object":
                stripped = data[col].astype(str).str.strip()
                converted = pd.to_numeric(stripped, errors="ignore")
                data[col] = converted

        return data

    except Exception as e:
        st.error(f"Failed to load the file: {e}")
        return None


# ============================================================
# HELPER FUNCTIONS: DATA PROFILING / TASK DETECTION
# ============================================================
def get_dataset_profile(data: pd.DataFrame) -> Dict[str, float]:
    """
    Return a compact dataset quality profile for display.
    """
    total_cells = data.shape[0] * data.shape[1] if not data.empty else 0
    missing_cells = int(data.isna().sum().sum()) if total_cells > 0 else 0
    duplicate_rows = int(data.duplicated().sum()) if not data.empty else 0

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in data.columns if c not in numeric_cols]

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

    Rule of thumb:
    - Numeric target with many unique values -> regression
    - Otherwise -> classification
    """
    y_non_null = y_series.dropna()

    if y_non_null.empty:
        return "classification"

    if pd.api.types.is_numeric_dtype(y_non_null):
        return "regression" if y_non_null.nunique() > max(15, len(y_non_null) * 0.05) else "classification"

    return "classification"


def find_valid_targets(data: pd.DataFrame) -> List[str]:
    """
    Determine which columns are usable as target columns.

    We exclude:
    - unnamed/placeholder columns
    - columns with very few non-null values
    - columns with only one unique value
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

    Instead of dropping all datetime columns, we try to convert them into:
    - year
    - month
    - day
    - day of week

    This gives the model a chance to learn from time information.
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
    """
    X = X.copy()
    dropped_summary = {
        "high_missing": [],
        "constant": [],
    }

    high_missing_cols = X.columns[X.isna().mean() > 0.60].tolist()
    if high_missing_cols:
        dropped_summary["high_missing"] = high_missing_cols
        X = X.drop(columns=high_missing_cols)

    constant_cols = X.columns[X.nunique(dropna=False) <= 1].tolist()
    if constant_cols:
        dropped_summary["constant"] = constant_cols
        X = X.drop(columns=constant_cols)

    return X, dropped_summary


def standardize_mixed_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert mixed-type columns to strings to avoid downstream transformer issues.
    """
    X = X.copy()

    for col in X.columns:
        try:
            if X[col].dropna().apply(type).nunique() > 1:
                X[col] = X[col].astype(str)
        except Exception:
            X[col] = X[col].astype(str)

    return X


def clean_numeric_like_series(series: pd.Series) -> pd.Series:
    """
    Clean a potentially numeric text column before conversion.

    Handles common patterns like:
    - commas: 1,200
    - currency: $1200
    - percentages: 45%
    - extra spaces

    This function is intentionally lightweight to avoid slowing the app.
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
                "none": np.nan,
                "null": np.nan,
            }
        )
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def infer_feature_types(X: pd.DataFrame, numeric_threshold: float = 0.85):
    """
    Infer numeric vs categorical columns more intelligently.

    Why this is needed:
    Some datasets store numbers as text, for example:
    - "1000"
    - "$2,500"
    - "45%"

    Logic:
    - keep already-numeric columns as numeric
    - for object columns, attempt numeric conversion
    - if enough non-null values convert successfully, treat as numeric
    - otherwise keep as categorical

    The threshold is set conservatively to avoid misclassifying
    genuinely categorical columns.
    """
    X = X.copy()
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        series = X[col]

        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        if series.dtype == "object":
            numeric_candidate = clean_numeric_like_series(series)

            non_null_original = series.notna().sum()

            # If there are no non-null values, keep as categorical
            if non_null_original == 0:
                categorical_cols.append(col)
                continue

            converted_ratio = numeric_candidate.notna().sum() / non_null_original

            if converted_ratio >= numeric_threshold:
                X[col] = numeric_candidate
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            categorical_cols.append(col)

    return X, numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame):
    """
    Build a preprocessing pipeline for numeric and categorical columns.

    Numeric:
    - median imputation
    - standard scaling

    Categorical:
    - most frequent imputation
    - one-hot encoding

    This version uses smarter feature-type inference so numeric-looking
    text columns are not incorrectly forced into categorical encoding.
    """
    X = X.copy()

    # Try to extract features from datetime-like columns
    X = convert_datetime_features(X)

    # Remove obviously problematic columns
    X, dropped_summary = drop_problematic_columns(X)

    # Standardize mixed-type columns so encoders behave consistently
    X = standardize_mixed_columns(X)

    # Smarter numeric/categorical detection
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
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    return X, preprocessor, numeric_cols, categorical_cols, dropped_summary


# ============================================================
# HELPER FUNCTIONS: MODELS / PIPELINES / TUNING
# ============================================================
def get_models(task_type: str) -> Dict[str, object]:
    """
    Return candidate models based on the task type.
    """
    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=10000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "SVM": SVC(probability=False),
            "KNN": KNeighborsClassifier(),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=200,
            )

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMClassifier(random_state=42, n_estimators=200)

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "SVM": SVR(),
            "KNN": KNeighborsRegressor(),
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(random_state=42, n_estimators=200)

        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMRegressor(random_state=42, n_estimators=200)

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


def get_param_grid(model_name: str) -> Dict[str, List]:
    """
    Hyperparameter search spaces for supported models.
    """
    if model_name == "Random Forest":
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5],
        }

    if model_name == "XGBoost":
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
        }

    if model_name == "LightGBM":
        return {
            "model__n_estimators": [100, 200],
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.05, 0.1],
        }

    return {}


def run_grid_search(model_name: str, pipeline: Pipeline, X_train, y_train, task_type: str) -> Pipeline:
    """
    Run GridSearchCV on supported models.
    If the model is not supported or tuning fails, return the original pipeline.
    """
    param_grid = get_param_grid(model_name)

    if not param_grid:
        st.info(f"No tuning grid defined for {model_name}. Using default settings.")
        return pipeline

    try:
        scoring = "accuracy" if task_type == "classification" else "neg_root_mean_squared_error"

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        st.success(f"Best parameters for {model_name}: {grid.best_params_}")
        return grid.best_estimator_

    except Exception as e:
        st.warning(f"Grid search skipped for {model_name}: {e}")
        return pipeline


def evaluate_models(X_train, y_train, models: Dict[str, object], preprocessor, task_type: str) -> pd.DataFrame:
    """
    Evaluate candidate models using cross-validation.

    For classification:
    - score = mean accuracy

    For regression:
    - score = mean RMSE
      (lower is better)

    Returns a DataFrame for easier sorting and display.
    """
    rows = []

    for name, model in models.items():
        try:
            pipeline = build_pipeline(preprocessor, clone(model))

            if task_type == "classification":
                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=None,
                )
                rows.append({
                    "Model": name,
                    "CV Score": float(np.mean(cv_scores)),
                    "Score Type": "Accuracy",
                })
            else:
                cv_scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=None,
                )
                rows.append({
                    "Model": name,
                    "CV Score": float(-np.mean(cv_scores)),
                    "Score Type": "RMSE",
                })

        except Exception:
            continue

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        ascending = task_type == "regression"
        results_df = results_df.sort_values("CV Score", ascending=ascending).reset_index(drop=True)

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

    ax.bar(results_df["Model"], results_df["CV Score"])
    ax.set_title("Model Comparison", fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("CV Accuracy" if task_type == "classification" else "CV RMSE")
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
        numeric_target = pd.to_numeric(y_raw, errors="coerce").dropna()
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

    For classification, stratification is helpful, but it can fail when:
    - one or more classes have too few examples

    This function falls back gracefully when needed.
    """
    if task_type == "classification":
        try:
            y_series = pd.Series(y)
            min_class_count = y_series.value_counts().min()

            if min_class_count >= 2:
                return train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )

        except Exception:
            pass

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


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
        unsafe_allow_html=True
    )

    st.sidebar.title("Workflow")
    st.sidebar.write("1. Upload data")
    st.sidebar.write("2. Choose target and features")
    st.sidebar.write("3. Run model comparison")
    st.sidebar.write("4. Review results and download the best pipeline")

    with st.sidebar.expander("Supported file types", expanded=False):
        st.write("CSV, XLSX, JSON")

    with st.sidebar.expander("What this app does", expanded=False):
        st.write(
            """
            - Cleans messy headers
            - Handles missing values
            - Detects numeric-like text columns
            - Encodes categorical columns
            - Compares several ML models
            - Evaluates the best model
            - Exports the trained pipeline
            """
        )

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "json"]
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
            horizontal=True
        )

    available_features = [col for col in data.columns if col != target_column]

    selected_features = st.multiselect(
        "Select feature columns",
        available_features,
        default=available_features
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
                y = pd.to_numeric(y_raw, errors="coerce")
                valid_idx = y.dropna().index
                X = X.loc[valid_idx].copy()
                y = y.loc[valid_idx].copy()

                if y.empty:
                    st.error("The selected target could not be used for regression.")
                    return

            else:
                y = y_raw.astype(str)
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            X, preprocessor, numeric_cols, categorical_cols, dropped_summary = build_preprocessor(X)

            if X.empty or len(X.columns) == 0:
                st.error("No valid feature columns remained after cleaning.")
                return

            st.subheader("Feature Preparation Summary")
            prep_col1, prep_col2, prep_col3, prep_col4 = st.columns(4)
            prep_col1.metric("Features Used", len(X.columns))
            prep_col2.metric("Numeric Features", len(numeric_cols))
            prep_col3.metric("Categorical Features", len(categorical_cols))
            prep_col4.metric(
                "Dropped Columns",
                len(dropped_summary["high_missing"]) + len(dropped_summary["constant"])
            )

            with st.expander("See detected feature types", expanded=False):
                st.write("Numeric columns:", numeric_cols if numeric_cols else "None")
                st.write("Categorical columns:", categorical_cols if categorical_cols else "None")

            with st.expander("See dropped columns", expanded=False):
                st.write("Dropped for high missingness (> 60%):", dropped_summary["high_missing"] or "None")
                st.write("Dropped for being constant:", dropped_summary["constant"] or "None")

            X_train, X_test, y_train, y_test = safe_train_test_split(X, y, task_type)

            models = get_models(task_type)
            results_df = evaluate_models(X_train, y_train, models, preprocessor, task_type)

            if results_df.empty:
                st.error("All models failed during evaluation.")
                return

            st.subheader("Model Comparison")
            st.dataframe(results_df, use_container_width=True)
            plot_model_results(results_df, task_type)

            if task_type == "classification":
                best_model_name = results_df.iloc[0]["Model"]
            else:
                best_model_name = results_df.iloc[0]["Model"]

            st.success(f"Best model selected: {best_model_name}")

            best_pipeline = build_pipeline(preprocessor, clone(models[best_model_name]))

            if tune_model:
                best_pipeline = run_grid_search(
                    best_model_name,
                    best_pipeline,
                    X_train,
                    y_train,
                    task_type
                )

            best_pipeline.fit(X_train, y_train)
            y_pred = best_pipeline.predict(X_test)

            st.subheader("Holdout Test Performance")

            if task_type == "classification":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy", f"{acc:.3f}")
                mc2.metric("Precision", f"{prec:.3f}")
                mc3.metric("Recall", f"{rec:.3f}")
                mc4.metric("F1 Score", f"{f1:.3f}")

                st.write("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                st.dataframe(pd.DataFrame(cm), use_container_width=True)

                if label_encoder is not None:
                    with st.expander("Encoded class labels", expanded=False):
                        st.write(pd.DataFrame({
                            "Encoded Value": range(len(label_encoder.classes_)),
                            "Original Label": label_encoder.classes_
                        }))

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
                mime="application/octet-stream"
            )

            st.success("Done. Your best model has been trained and packaged for download.")


# ============================================================
# APP ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
# streamlit run automated_ml_tool.py
