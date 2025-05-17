#pip install --upgrade pip
#streamlit run automated_ml_tool.py
#pip install openpyxl

# --- Import necessary libraries ---
import pandas as pd  # for data handling
import numpy as np  # for numerical operations
import streamlit as st  # for building the web UI
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  # for model training and tuning
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # for feature scaling and encoding
from sklearn.impute import SimpleImputer  # for handling missing data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # tree-based models
from sklearn.linear_model import LogisticRegression, LinearRegression  # linear models
from sklearn.svm import SVC, SVR  # support vector machines
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # k-nearest neighbors
from sklearn.base import is_classifier  # to check model type
import joblib  # for saving models
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for nice plots
import xgboost as xgb  # XGBoost model
import lightgbm as lgb  # LightGBM model

# --- Ensure column names in a dataset are unique ---
def make_column_names_unique(columns):
    counts = {}
    new_columns = []
    for col in columns:
        if col not in counts:
            counts[col] = 0
            new_columns.append(col)
        else:
            counts[col] += 1
            new_columns.append(f"{col}_{counts[col]}")
    return new_columns

# --- Load and clean uploaded dataset file ---
def load_and_clean_data(uploaded_file):
    try:
        # Load file based on extension
        if uploaded_file.name.endswith('.csv'):
            raw_data = pd.read_csv(uploaded_file, header=None)
        elif uploaded_file.name.endswith('.xlsx'):
            raw_data = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
        elif uploaded_file.name.endswith('.json'):
            raw_data = pd.read_json(uploaded_file)
            if isinstance(raw_data, dict):  # convert nested dicts to dataframe
                raw_data = pd.json_normalize(raw_data)
        else:
            raise ValueError("Unsupported file format.")

        # Handle list format and ensure tabular structure
        if isinstance(raw_data, list):
            raw_data = pd.DataFrame(raw_data)
        elif not isinstance(raw_data, pd.DataFrame):
            raise ValueError("Uploaded file is not in a recognizable tabular format.")

        # Detect header row heuristically
        try:
            if "date" in str(raw_data.iloc[2, 0]).strip().lower():
                header_row = 2
            else:
                header_row = 0
        except:
            header_row = 0

        # If many bad signals, adjust header_row
        sample_header = raw_data.iloc[header_row].astype(str).str.lower()
        bad_signals = ["unnamed", "nan", "", "null"]
        signal_count = sample_header.apply(lambda val: any(sig in val for sig in bad_signals)).sum()
        if signal_count > (0.5 * len(sample_header)):
            header_row = raw_data.notna().sum(axis=1).idxmax()

        # Extract data and format columns
        data = raw_data[header_row + 1:].copy()
        data.columns = raw_data.iloc[header_row]
        data.columns = [str(col).strip().replace("\n", " ") for col in data.columns]
        data.columns = make_column_names_unique(data.columns)
        data = data.dropna(axis=1, how='all').dropna(axis=0, how='all').reset_index(drop=True)

        return data

    except Exception as e:
        st.error(f"Failed to load and clean data: {e}")
        return None

# --- Try to convert all columns to numeric where possible ---
def safe_convert_numeric(df):
    try:
        return df.apply(pd.to_numeric, errors='ignore')
    except Exception:
        return df

# --- Preprocess data: clean, impute, encode, and scale ---
def preprocess_data(df, target_column, apply_log_transform=True):
    try:
        df = df.copy()

        # Extract target column and convert to numeric
        y = pd.to_numeric(df[target_column], errors='coerce')
        X = df.drop(columns=[target_column])

        # Remove rows where target is missing
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        # Clip extreme outliers
        y = y.clip(upper=y.quantile(0.99))

        # Filter again post clipping
        X = X.loc[y.notnull()]
        y = y.loc[y.notnull()]

        # Remove columns with >50% missing or constant values
        X = X.loc[:, X.isnull().mean() < 0.5]
        X = X.loc[:, X.nunique(dropna=False) > 1]

        # Convert eligible columns to numeric
        X = safe_convert_numeric(X)

        # Remove datetime columns
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            X = X.drop(columns=datetime_cols)

        # Split features into numeric and categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Catch mixed types in columns and treat as categorical
        for col in X.columns:
            if X[col].apply(type).nunique() > 1:
                X[col] = X[col].astype(str)
                if col not in categorical_cols:
                    categorical_cols.append(col)

        # Impute missing numeric values
        if numeric_cols:
            imputer_num = SimpleImputer(strategy='mean')
            X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])

        # Impute and one-hot encode categorical variables
        if categorical_cols:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols].astype(str))
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(X[categorical_cols])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
            X = X.drop(columns=categorical_cols)
            X = pd.concat([X, encoded_df], axis=1)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_final = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        y = y.loc[X_final.index]

        # Log-transform skewed target
        skewness = y.skew()
        if abs(skewness) > 3 and apply_log_transform:
            st.warning(f"⚠️ Target is highly skewed (skew = {skewness:.2f}). Applying log1p transform.")
            y = np.log1p(y)
            valid_idx = y.replace([np.inf, -np.inf], np.nan).dropna().index
            X_final = X_final.loc[valid_idx]
            y = y.loc[valid_idx]

        st.write(f"✅ Using {X_final.shape[0]} rows and {X_final.shape[1]} features.")
        return X_final, y

    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None

# --- Return appropriate model list based on task type ---
def get_models(task_type):
    if task_type == 'classification':
        return [
            ('Logistic Regression', LogisticRegression(max_iter=10000)),
            ('Random Forest', RandomForestClassifier(n_estimators=100)),
            ('SVM', SVC()),
            ('KNN', KNeighborsClassifier()),
            ('XGBoost', xgb.XGBClassifier()),
            ('LightGBM', lgb.LGBMClassifier())
        ]
    else:
        return [
            ('Linear Regression', LinearRegression()),
            ('Random Forest', RandomForestRegressor(n_estimators=100)),
            ('SVM', SVR()),
            ('KNN', KNeighborsRegressor()),
            ('XGBoost', xgb.XGBRegressor()),
            ('LightGBM', lgb.LGBMRegressor())
        ]

# --- Optionally tune model hyperparameters using GridSearch ---
def run_grid_search(model_name, model, X_train, y_train):
    param_grid = {}

    # Set grid search parameters for supported models
    if model_name == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10]
        }
    elif model_name == "XGBoost":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6]
        }
    elif model_name == "LightGBM":
        param_grid = {
            'n_estimators': [50, 100],
            'num_leaves': [31, 64]
        }

    # If model is unsupported, return default
    if not param_grid:
        return model

    try:
        st.write(f"🔍 Running Grid Search for {model_name}...")
        scoring = 'accuracy' if is_classifier(model) else 'neg_mean_squared_error'
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        st.success(f"✅ Best Params for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        st.warning(f"Grid search failed for {model_name}: {e}")
        return model

# --- Evaluate models with cross-validation ---
def evaluate_models(X, y, models, task_type):
    results = {}
    for name, model in models:
        try:
            score = (cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
                     if task_type == 'classification'
                     else -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean())
            results[name] = score
        except Exception:
            results[name] = float('nan')
    return {k: v for k, v in results.items() if not np.isnan(v)}

# --- Plot evaluation results as bar chart ---
def plot_results(results):
    if not results:
        st.warning("No valid models evaluated.")
        return
    try:
        names, scores = zip(*results.items())
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(names), y=list(scores))
        plt.xticks(rotation=45)
        plt.title("Model Performance")
        plt.tight_layout()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Plotting failed: {e}")

# --- Main Streamlit app ---
def streamlit_app():
    st.title("📊 AutoML Tool")
    st.write("Upload a dataset and let the tool do the work.")

    uploaded_file = st.file_uploader("Upload Your Dataset", type=["csv", "xlsx", "json"])
    if uploaded_file:
        data = load_and_clean_data(uploaded_file)
        if data is not None:
            st.write("Preview:", data.head())

            # Suggest valid targets (ignore columns with mostly nulls or constant values)
            valid_targets = [str(col) for col in data.columns
                             if isinstance(col, str) and not str(col).lower().startswith('unnamed')
                             and data[col].notnull().sum() > 10 and data[col].nunique() > 1]

            if not valid_targets:
                st.error("No usable target columns found.")
                return

            # Target and feature selection
            target_column = st.selectbox("Select the target column:", options=valid_targets)
            available_features = [col for col in data.columns if col != target_column]
            selected_features = st.multiselect("Select features:", available_features)

            if not selected_features:
                st.warning("Please select at least one feature.")
                return

            # Classification or regression toggle
            task_type = st.radio("Choose problem type", ["Classification", "Regression"],
                                 index=0 if data[target_column].nunique() <= 20 else 1)

            # Run AutoML pipeline
            if st.button("Run AutoML"):
                X_processed, y = preprocess_data(data[selected_features + [target_column]], target_column)

                if X_processed is None or y is None:
                    st.warning("Preprocessing failed.")
                    return

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

                st.write("Evaluating models...")
                models = get_models(task_type.lower())
                results = evaluate_models(X_train, y_train, models, task_type.lower())

                if not results:
                    st.error("All models failed. Please check your dataset.")
                    return

                st.write("Results:", results)
                plot_results(results)

                # Final model selection and training
                best_model_name = max(results, key=results.get)
                st.success(f"Best model: {best_model_name}")
                model = dict(models)[best_model_name]
                model = run_grid_search(best_model_name, model, X_train, y_train)

                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Regression metrics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.write(f"📉 RMSE: {rmse:.2f}")
                    st.write(f"📊 MAE: {mae:.2f}")
                    st.write(f"📈 R² Score: {r2:.2f}")

                    # Save and download trained model
                    joblib.dump(model, "best_model.pkl")
                    with open("best_model.pkl", "rb") as f:
                        st.download_button("Download Trained Model", data=f, file_name="best_model.pkl")
                except Exception as e:
                    st.error(f"Training error: {e}")

# --- Run the app ---
if __name__ == "__main__":
    streamlit_app()
#Run in terminal: streamlit run ml_tool_app.py
