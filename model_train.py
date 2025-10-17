# src/model_train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data_prep import load_and_process  # ✅ fixed import to match folder structure

def train_model(data_path):
    """
    Load, preprocess, and train a Random Forest model.

    Args:
        data_path (str): Path to the raw sales CSV data.

    Returns:
        model: trained Random Forest model
        X_test, y_test: test data
        y_pred: predictions
    """
    # ✅ Load and preprocess the data
    X, y, df_processed = load_and_process(data_path)

    # ✅ Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # good for time-series
    )

    # ✅ Initialize model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # ✅ Train the model
    model.fit(X_train, y_train)

    # ✅ Predict on test data
    y_pred = model.predict(X_test)

    # ✅ Evaluate
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # manually take sqrt
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model trained successfully! RMSE: {rmse:.2f}, R²: {r2:.2f}")

    return model, X_test, y_test, y_pred
