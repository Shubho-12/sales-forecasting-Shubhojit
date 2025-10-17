# src/predict.py

import os
import joblib
import pandas as pd
from data_prep import load_and_process

# Define model path
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_model.pkl'))

# Check if model exists
if not os.path.exists(model_path):
    print("Model not found. Train your model first!")
    exit()

# Load model
model = joblib.load(model_path)
print(f"Model loaded from: {model_path}")

# Load and process new data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'sales.csv'))
X, y, df_processed = load_and_process(data_path)

# Predict
y_pred = model.predict(X)

# Combine results
df_processed['Predicted_Sales'] = y_pred

# Save predictions
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sales_predictions.csv'))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_processed.to_csv(output_path, index=False)

print(f"Predictions saved to: {output_path}")

# Show comparison of actual vs predicted
print("Sample comparison (Actual vs Predicted):")
print(df_processed[['units_sold', 'Predicted_Sales']].head())
