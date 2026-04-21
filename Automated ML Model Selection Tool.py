#pip install --upgrade pip
#streamlit run automated_ml_tool.py
#pip install openpyxl

# --- Import necessary libraries ---
# Import standard libraries
import io                  # lets us create an in-memory file for downloading the trained model
import joblib              # used to save the trained pipeline/model
import numpy as np         # numerical operations
import pandas as pd        # data handling and cleaning
import streamlit as st     # web app interface
import matplotlib.pyplot as plt  # plotting charts

# Import sklearn utilities and models
from sklearn.base import is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Try to import XGBoost
# If it is not installed, the program will continue without it
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM
# If it is not installed, the program will continue without it
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


# Configure the Streamlit page settings
st.set_page_config(
    page_title="Free AutoML Tool",
    page_icon="📊",
    layout="wide"
)


# Function to make duplicate or messy column names unique and cleaner
def make_column_names_unique(columns):
    counts = {}          # stores count of repeated column names
    new_columns = []     # stores cleaned/unique column names

    for col in columns:
        # Convert column name to string, remove extra spaces/newlines
        col = str(col).strip().replace("\n", " ")

        # If column name appears for the first time, keep it
        if col not in counts:
            counts[col] = 0
            new_columns.append(col)
        else:
            # If duplicate exists, add suffix like _1, _2, etc.
            counts[col] += 1
            new_columns.append(f"{col}_{counts[col]}")

    return new_columns


# Function to load and clean uploaded dataset
def load_and_clean_data(uploaded_file):
    try:
        # Convert file name to lowercase for easier extension checking
        filename = uploaded_file.name.lower()

        # Read CSV file
        if filename.endswith(".csv"):
            raw_data = pd.read_csv(uploaded_file, header=None)

        # Read Excel file
        elif filename.endswith(".xlsx"):
            raw_data = pd.read_excel(uploaded_file, engine="openpyxl", header=None)

        # Read JSON file
        elif filename.endswith(".json"):
            raw_json = pd.read_json(uploaded_file)
            raw_data = pd.json_normalize(raw_json) if isinstance(raw_json, dict) else raw_json

        # Reject unsupported formats
        else:
            raise ValueError("Unsupported file format. Please upload CSV, XLSX, or JSON.")

        # Ensure the loaded data is a DataFrame
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        # Try to detect which row should be treated as the header
        header_row = 0
        try:
            sample = raw_data.iloc[0:3].fillna("").astype(str)

            # Check first 3 rows and pick the one that looks most like real headers
            for i in range(min(3, len(raw_data))):
                row = sample.iloc[i]

                # Bad header signals: blank values, unnamed, nan, null
                bad_signals = row.str.lower().isin(["unnamed", "nan", "null", ""])

                # If less than half are bad signals, use this row as header
                if bad_signals.mean() < 0.5:
                    header_row = i
                    break
        except Exception:
            header_row = 0

        # Separate actual data from header row
        data = raw_data[header_row + 1:].copy()

        # Set cleaned unique column names
        data.columns = make_column_names_unique(raw_data.iloc[header_row])

        # Remove fully empty rows and columns
        data = data.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

        return data

    except Exception as e:
        # Show error in Streamlit if file loading fails
        st.error(f"Failed to load the file: {e}")
        return None


# Function to automatically detect whether task is classification or regression
def auto_detect_task(y_series):
    y_non_null = y_series.dropna()

    # If target column is empty, default to classification
    if y_non_null.empty:
        return "classification"

    # If target is numeric and has many unique values, treat as regression
    if pd.api.types.is_numeric_dtype(y_non_null):
        return "regression" if y_non_null.nunique() > 20 else "classification"

    # Otherwise treat as classification
    return "classification"


# Function to build preprocessing pipeline for features
def build_preprocessor(X):
    X = X.copy()

    # Drop columns with more than 50% missing values
    X = X.loc[:, X.isnull().mean() < 0.5]

    # Drop constant columns (columns with only one unique value)
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]

    # Convert mixed-type columns into strings so encoding can handle them
    for col in X.columns:
        try:
            if X[col].apply(type).nunique() > 1:
                X[col] = X[col].astype(str)
        except Exception:
            X[col] = X[col].astype(str)

    # Drop datetime columns because they are not directly handled here
    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    if datetime_cols:
        X = X.drop(columns=datetime_cols)

    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    # Preprocessing steps for numeric columns:
    # 1. Fill missing values with mean
    # 2. Scale values
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing steps for categorical columns:
    # 1. Fill missing values with most frequent value
    # 2. Convert categories into numerical one-hot encoded format
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine numeric and categorical preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    return X, preprocessor, numeric_cols, categorical_cols


# Function to return a dictionary of models depending on task type
def get_models(task_type):
    models = {}

    if task_type == "classification":
        # Classification models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=10000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
        }

        # Add XGBoost classifier if available
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                eval_metric="logloss",
                random_state=42
            )

        # Add LightGBM classifier if available
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMClassifier(random_state=42)

    else:
        # Regression models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "SVM": SVR(),
            "KNN": KNeighborsRegressor(),
        }

        # Add XGBoost regressor if available
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(random_state=42)

        # Add LightGBM regressor if available
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMRegressor(random_state=42)

    return models


# Function to combine preprocessing and model into one pipeline
def build_pipeline(preprocessor, model):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


# Function to run Grid Search hyperparameter tuning for supported models
def run_grid_search(model_name, pipeline, X_train, y_train, task_type):
    param_grid = {}

    # Define tuning parameters depending on model
    if model_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 5, 10]
        }

    elif model_name == "XGBoost":
        param_grid = {
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 6]
        }

    elif model_name == "LightGBM":
        param_grid = {
            "model__n_estimators": [50, 100],
            "model__num_leaves": [31, 64]
        }

    # If no parameter grid exists, return original pipeline unchanged
    if not param_grid:
        return pipeline

    try:
        # Use accuracy for classification and negative MSE for regression
        scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring=scoring,
            n_jobs=-1
        )

        # Train grid search
        grid.fit(X_train, y_train)

        # Show best parameters found
        st.success(f"Best parameters for {model_name}: {grid.best_params_}")

        return grid.best_estimator_

    except Exception as e:
        # If tuning fails, continue with original pipeline
        st.warning(f"Grid search skipped for {model_name}: {e}")
        return pipeline


# Function to evaluate all models using cross-validation
def evaluate_models(X_train, y_train, models, preprocessor, task_type):
    results = {}

    for name, model in models.items():
        try:
            # Build full pipeline for each model
            pipeline = build_pipeline(preprocessor, model)

            # For classification: use accuracy
            if task_type == "classification":
                score = cross_val_score(
                    pipeline, X_train, y_train, cv=5, scoring="accuracy"
                ).mean()

            # For regression: use mean squared error
            else:
                score = -cross_val_score(
                    pipeline, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
                ).mean()

            # Store average CV score
            results[name] = score

        except Exception:
            # Skip model if it fails
            continue

    return results


# Function to visualize model results as a bar chart
def plot_results(results, task_type):
    if not results:
        st.warning("No valid models could be evaluated.")
        return

    names = list(results.keys())
    scores = list(results.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, scores)
    ax.set_title("Model Performance")
    ax.set_ylabel("Accuracy" if task_type == "classification" else "CV MSE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    # Display chart in Streamlit
    st.pyplot(fig)


# Main application function
def main():
    # App title and short instruction
    st.title("📊 Free AutoML Tool")
    st.write(
        "Upload a dataset, choose your target column, and let the app compare machine learning models for you."
    )

    # Expandable instructions section
    with st.expander("How it works"):
        st.write(
            """
            - Upload a CSV, XLSX, or JSON file
            - Choose the target column
            - Select which features to use
            - Pick classification or regression
            - Run AutoML and compare models
            - Download the best trained pipeline
            """
        )

    # Upload widget
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "json"]
    )

    # Stop if no file is uploaded
    if not uploaded_file:
        return

    # Load and clean dataset
    data = load_and_clean_data(uploaded_file)

    # Stop if data failed to load
    if data is None or data.empty:
        st.error("The uploaded file could not be processed.")
        return

    # Show preview of dataset
    st.subheader("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    # Identify valid columns that can serve as target columns
    valid_targets = [
        col for col in data.columns
        if not str(col).lower().startswith("unnamed")
        and data[col].notnull().sum() > 10
        and data[col].nunique(dropna=True) > 1
    ]

    # Stop if no target column is suitable
    if not valid_targets:
        st.error("No usable target columns were found.")
        return

    # Create two side-by-side input columns
    col1, col2 = st.columns(2)

    with col1:
        # Let user select the target column
        target_column = st.selectbox("Select target column", valid_targets)

    with col2:
        # Automatically suggest classification or regression
        detected_task = auto_detect_task(data[target_column])

        # Let user confirm or change problem type
        task_type = st.radio(
            "Problem type",
            ["classification", "regression"],
            index=0 if detected_task == "classification" else 1,
            horizontal=True
        )

    # Feature columns = everything except target
    available_features = [col for col in data.columns if col != target_column]

    # Let user choose which feature columns to use
    selected_features = st.multiselect(
        "Select features",
        available_features,
        default=available_features[: min(8, len(available_features))]
    )

    # Optional checkbox for hyperparameter tuning
    tune_model = st.checkbox("Tune supported models with Grid Search", value=False)

    # Stop if no feature selected
    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    # Run AutoML when button is clicked
    if st.button("Run AutoML", type="primary"):
        with st.spinner("Preparing data and evaluating models..."):
            # Keep only selected features and target
            df_model = data[selected_features + [target_column]].copy()

            # Separate input features (X) and target (y)
            X = df_model.drop(columns=[target_column])
            y_raw = df_model[target_column]

            # Drop rows where target is missing
            valid_idx = y_raw.dropna().index
            X = X.loc[valid_idx].copy()
            y_raw = y_raw.loc[valid_idx].copy()

            # For regression, convert target to numeric
            if task_type == "regression":
                y = pd.to_numeric(y_raw, errors="coerce")
                valid_idx = y.dropna().index
                X = X.loc[valid_idx].copy()
                y = y.loc[valid_idx].copy()

                # Stop if no valid regression target remains
                if y.empty:
                    st.error("Target column could not be used for regression.")
                    return

            else:
                # For classification, convert target labels into encoded numbers
                y = y_raw.astype(str)
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            # Build cleaned feature set and preprocessing pipeline
            X, preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

            # Stop if no valid features remain
            if X.empty or len(X.columns) == 0:
                st.error("No valid feature columns remained after cleaning.")
                return

            # Stratify only for classification
            stratify_y = y if task_type == "classification" else None

            # Split dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_y
            )

            # Load model options
            models = get_models(task_type)

            # Evaluate all models
            results = evaluate_models(X_train, y_train, models, preprocessor, task_type)

            # Stop if all models fail
            if not results:
                st.error("All models failed during evaluation.")
                return

            # Show model comparison results
            st.subheader("Model Comparison")
            st.write(results)
            plot_results(results, task_type)

            # Pick the best model
            # For classification: highest accuracy is best
            # For regression: lowest MSE is best
            best_model_name = max(results, key=results.get) if task_type == "classification" else min(results, key=results.get)
            st.success(f"Best model: {best_model_name}")

            # Build pipeline with best model
            best_pipeline = build_pipeline(preprocessor, models[best_model_name])

            # Optionally tune best model
            if tune_model:
                best_pipeline = run_grid_search(
                    best_model_name,
                    best_pipeline,
                    X_train,
                    y_train,
                    task_type
                )

            # Train best pipeline on training data
            best_pipeline.fit(X_train, y_train)

            # Make predictions on test set
            y_pred = best_pipeline.predict(X_test)

            # Show test set metrics
            st.subheader("Test Metrics")

            if task_type == "classification":
                # Classification performance metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                # Display metrics in columns
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("Precision", f"{prec:.3f}")
                c3.metric("Recall", f"{rec:.3f}")
                c4.metric("F1 Score", f"{f1:.3f}")

                # Show confusion matrix
                st.write("Confusion Matrix")
                st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

            else:
                # Regression performance metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Display metrics in columns
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{rmse:.3f}")
                c2.metric("MAE", f"{mae:.3f}")
                c3.metric("R²", f"{r2:.3f}")

            # Save trained pipeline and useful metadata
            artifact = {
                "pipeline": best_pipeline,
                "task_type": task_type,
                "target_column": target_column,
                "selected_features": selected_features,
            }

            # Also save label encoder for classification problems
            if task_type == "classification":
                artifact["label_encoder"] = label_encoder

            # Save artifact into memory buffer for download
            buffer = io.BytesIO()
            joblib.dump(artifact, buffer)
            buffer.seek(0)

            # Download button for trained model
            st.download_button(
                "Download Trained Pipeline",
                data=buffer,
                file_name="best_automl_pipeline.pkl",
                mime="application/octet-stream"
            )


# Run the app
if __name__ == "__main__":
    main()
#Run in terminal: streamlit run automated_ml_tool.py
