# src/data_prep.py

import pandas as pd
import numpy as np


def load_and_process(data):
    """
    Load raw sales data and perform feature engineering.

    Args:
        data: str (file path) or pandas DataFrame

    Returns:
        X: Features DataFrame
        y: Target Series (or None if 'units_sold' not present)
        df: Processed DataFrame with extra columns
    """
    # If input is a path, read CSV
    if isinstance(data, str):
        try:
            df = pd.read_csv(data, encoding='utf-16')  # Try utf-16
        except UnicodeError:
            df = pd.read_csv(data, encoding='utf-8-sig')  # fallback
    else:
        df = data.copy()  # assume DataFrame

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # Time-based features
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['dayofweek'] = df['date'].dt.dayofweek
    df['trend'] = np.arange(len(df))

    # Lag features
    if 'units_sold' in df.columns:
        df['lag_1'] = df['units_sold'].shift(1).bfill()
        df['lag_3'] = df['units_sold'].shift(3).bfill()
        # Rolling mean
        df['rolling_mean_3'] = df['units_sold'].rolling(3).mean().bfill()
    else:
        # If no units_sold (future dataset), fill with zeros for lag/rolling
        df['lag_1'] = 0
        df['lag_3'] = 0
        df['rolling_mean_3'] = 0

    # Features
    feature_cols = ['trend', 'month', 'quarter', 'dayofweek', 'promo', 'holiday',
                    'lag_1', 'lag_3', 'rolling_mean_3']

    # Ensure all columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols]

    # Target
    y = df['units_sold'] if 'units_sold' in df.columns else None

    return X, y, df
