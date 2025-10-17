import joblib
import pandas as pd
import numpy as np
import os

# ===============================
# 1️⃣ Load Saved Model and Columns
# ===============================
def load_model():
    """Load trained model and feature columns."""
    model_path = os.path.join("models", "sales_forecast_model.pkl")
    columns_path = os.path.join("models", "X_train_columns.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model file not found! Train the model first.")
    if not os.path.exists(columns_path):
        raise FileNotFoundError("❌ X_train_columns.pkl not found! Run training again.")

    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
    return model, feature_columns


# ===============================
# 2️⃣ Create Input Sample
# ===============================
def create_input_sample():
    """
    Example: manually create one input data point for prediction.
    Modify these fields as per your dataset’s features.
    """
    sample_dict = {
        "month": 10,
        "dayofweek": 3,
        "trend": 120,
        "lag_1": 250,
        "rolling_mean_3": 245
    }
    return pd.DataFrame([sample_dict])


# ===============================
# 3️⃣ Predict Sales
# ===============================
def predict_sales(model, input_df, feature_columns):
    """
    Predict sales for the given input dataframe.
    Ensures column alignment with training features.
    """
    # Add any missing columns (for safety)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[feature_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)


# ===============================
# 4️⃣ Main Runner
# ===============================
if __name__ == "__main__":
    print("🚀 Loading model and columns...")
    model, feature_columns = load_model()

    print("✅ Model loaded successfully!\n")

    sample = create_input_sample()
    print("📊 Input sample:")
    print(sample)

    print("\n🔮 Making prediction...")
    predicted_sales = predict_sales(model, sample, feature_columns)

    print(f"\n💰 Predicted Sales: {predicted_sales}")
    print("\n✅ Done! Your model works perfectly end-to-end.")
