# Import all required libraries
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline

# Data Importing and Preprocessing
def load_data(file_path):
    """Load dataset from various file formats."""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def preprocess_data(data):
    """Preprocess data by handling missing values, encoding, and scaling."""
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_filled = imputer.fit_transform(data.select_dtypes(include=[np.number]))

    # Encode categorical variables (if any)
    categorical_cols = data.select_dtypes(include=[object]).columns
    encoder = OneHotEncoder(drop='first', sparse=False)
    if len(categorical_cols) > 0:
        encoded_data = encoder.fit_transform(data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        data = data.drop(categorical_cols, axis=1)
        data = pd.concat([data, encoded_df], axis=1)

    # Standardize numerical features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))

    return pd.DataFrame(data_scaled, columns=data.select_dtypes(include=[np.number]).columns), data_filled

def split_data(data, target_column):
    """Split data into train and test sets."""
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
def get_classification_models():
    """Return a list of classification models."""
    return [
        ('Logistic Regression', LogisticRegression(max_iter=10000)),
        ('Random Forest', RandomForestClassifier(n_estimators=100)),
        ('SVM', SVC()),
        ('KNN', KNeighborsClassifier()),
        ('XGBoost', xgb.XGBClassifier()),
        ('LightGBM', lgb.LGBMClassifier())
    ]

def get_regression_models():
    """Return a list of regression models."""
    return [
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor(n_estimators=100)),
        ('SVM', SVR()),
        ('KNN', KNeighborsRegressor()),
        ('XGBoost', xgb.XGBRegressor()),
        ('LightGBM', lgb.LGBMRegressor())
    ]

def evaluate_models(X_train, y_train, models, task_type='classification'):
    """Evaluate different models based on cross-validation scores."""
    results = {}
    for name, model in models:
        if task_type == 'classification':
            cv_results = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            results[name] = cv_results.mean()
        else:
            cv_results = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            results[name] = -cv_results.mean()
    return results

def tune_hyperparameters(model, X_train, y_train, param_grid, randomized_search=False):
    """Tune hyperparameters using GridSearchCV or RandomizedSearchCV."""
    if randomized_search:
        search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=5, random_state=42, n_jobs=-1)
    else:
        search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

# Save and Load Model
def save_model(model, filename='best_model.pkl'):
    """Save the trained model to a file."""
    joblib.dump(model, os.path.join('models', filename))

def load_model(filename='best_model.pkl'):
    """Load a saved model."""
    return joblib.load(filename)

# Visualization
def plot_results(results):
    """Visualize the model evaluation results."""
    model_names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=scores)
    plt.xticks(rotation=45, ha='right')
    plt.title("Model Evaluation Scores")
    plt.xlabel("Model")
    plt.ylabel("Score")
    st.pyplot(plt)  # Streamlit-specific plotting

# Streamlit Web Interface
def streamlit_app():
    st.title('Automated Machine Learning Model Selection Tool')

    # Upload file
    uploaded_file = st.file_uploader("Upload Your Dataset", type=["csv", "xlsx", "json"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Dataset loaded successfully!", data.head())

            target_column = st.text_input("Enter the target column name:")
            if target_column and target_column in data.columns:
                X_train, X_test, y_train, y_test = split_data(data, target_column)

                task_type = st.radio("Select task type", ('Classification', 'Regression')).lower()

                # Preprocess data
                X_train_processed, _ = preprocess_data(X_train)
                X_test_processed, _ = preprocess_data(X_test)

                # Select models
                models = get_classification_models() if task_type == 'classification' else get_regression_models()

                # Evaluate models
                st.write("Evaluating models...")
                results = evaluate_models(X_train_processed, y_train, models, task_type)

                st.write("Model performance:")
                for model_name, score in results.items():
                    st.write(f"{model_name}: {score:.4f}")

                # Visualize Results
                st.write("Visualizing evaluation scores...")
                plot_results(results)

                # Hyperparameter Tuning
                best_model_name = max(results, key=results.get)
                st.write(f"Best model: {best_model_name}")
                model = dict(models)[best_model_name]

                param_grid = {
                    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]},
                    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
                    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.3]},
                    'LightGBM': {'num_leaves': [31, 50], 'learning_rate': [0.01, 0.1, 0.3]}
                }

                best_model, best_params = tune_hyperparameters(model, X_train_processed, y_train,
                                                               param_grid.get(best_model_name, {}), randomized_search=True)
                st.write(f"Best hyperparameters for {best_model_name}: {best_params}")
                st.write("Training the final model...")

                # Train the best model
                best_model.fit(X_train_processed, y_train)
                save_model(best_model)

                st.write("Model trained and saved successfully!")
                st.download_button("Download the trained model", "best_model.pkl")

# Main
if __name__ == '__main__':
    streamlit_app()
