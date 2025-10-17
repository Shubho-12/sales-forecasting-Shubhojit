# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.data_prep import load_and_process
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Streamlit Config
# ------------------------------------------------------------
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📈 Sales Forecasting Dashboard")

# ------------------------------------------------------------
# Load Model & Columns
# ------------------------------------------------------------
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("models/sales_forecast_model.pkl")
    X_cols = joblib.load("models/X_train_columns.pkl")
    return model, X_cols

model, X_cols = load_model_and_columns()

# ------------------------------------------------------------
# Upload CSV
# ------------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload your company sales CSV", type=["csv"])

if uploaded_file is not None:
    # Load CSV as DataFrame
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    st.subheader("✅ Uploaded Data Preview")
    st.dataframe(df.head())

    # Optional: check for 'store' or 'product' columns
    group_cols = [c for c in ['store', 'product'] if c in df.columns]

    if group_cols:
        selected_group = st.selectbox(
            "Select a group to forecast (or Overall)",
            ["Overall"] + [f"{col} {val}" for col in group_cols for val in df[col].unique()]
        )
        if selected_group != "Overall":
            col_name, val = selected_group.split(" ", 1)
            df_group = df[df[col_name] == val].copy()
        else:
            df_group = df.copy()
    else:
        df_group = df.copy()

    # ------------------------------------------------------------
    # Data Preprocessing
    # ------------------------------------------------------------
    X, y, df_processed = load_and_process(df_group)

    st.subheader("📊 Processed Data Preview")
    rows_to_show = st.slider("Rows to preview", 5, len(df_processed), 10)
    st.dataframe(df_processed.head(rows_to_show))

    # ------------------------------------------------------------
    # Historical Predictions
    # ------------------------------------------------------------
    X_input = df_processed[X_cols]
    df_processed["Predicted"] = model.predict(X_input)

    st.subheader("📈 Historical Actual vs Predicted")
    st.line_chart(df_processed.set_index("date")[["units_sold", "Predicted"]])

    # Compute Metrics
    if "units_sold" in df_processed.columns:
        mae = mean_absolute_error(df_processed["units_sold"], df_processed["Predicted"])
        rmse = np.sqrt(mean_squared_error(df_processed["units_sold"], df_processed["Predicted"]))
        r2 = r2_score(df_processed["units_sold"], df_processed["Predicted"])

        col1, col2, col3 = st.columns(3)
        col1.metric("📏 MAE", f"{mae:.2f}")
        col2.metric("📉 RMSE", f"{rmse:.2f}")
        col3.metric("📊 R² Score", f"{r2:.2f}")

    # ------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------
    if hasattr(model, "feature_importances_"):
        st.subheader("🧠 Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": X_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance (Model Insights)")
        st.pyplot(fig)

    # ------------------------------------------------------------
    # Future Forecast
    # ------------------------------------------------------------
    forecast_months = st.number_input("🔮 Months to Forecast", min_value=1, max_value=36, value=6)

    if st.button("Generate Future Forecast"):
        def recursive_forecast(model, last_df, periods=6, freq='M', feature_cols=None,
                               value_col='units_sold', lags=(1, 3), rolling_windows=(3, 6)):

            # Fix FutureWarning
            if freq == 'M':
                freq = 'MS'

            last_date = pd.to_datetime(last_df['date']).max()
            future_dates = pd.date_range(
                start=last_date + pd.tseries.frequencies.to_offset(freq),
                periods=periods,
                freq=freq
            )

            full = pd.concat(
                [last_df[['date', value_col]], pd.DataFrame({'date': future_dates})],
                ignore_index=True, sort=False
            )
            full = full.sort_values('date').reset_index(drop=True)

            full['trend'] = np.arange(len(full))
            full['month'] = full['date'].dt.month
            full['dayofweek'] = full['date'].dt.dayofweek

            for lag in lags:
                full[f'lag_{lag}'] = full[value_col].shift(lag)
            for w in rolling_windows:
                full[f'rolling_mean_{w}'] = full[value_col].shift(1).rolling(w).mean()

            preds = []
            for idx in range(len(full)):
                if pd.notna(full.loc[idx, value_col]):
                    continue

                for col in feature_cols:
                    if col not in full.columns:
                        full[col] = 0

                X_row = full.loc[idx:idx, feature_cols]
                pred = model.predict(X_row)[0]
                preds.append(pred)
                full.at[idx, value_col] = pred

                for lag in lags:
                    full[f'lag_{lag}'] = full[value_col].shift(lag)
                for w in rolling_windows:
                    full[f'rolling_mean_{w}'] = full[value_col].shift(1).rolling(w).mean()

            future_rows = full[full['date'] > last_df['date'].max()].copy()
            future_rows['prediction'] = future_rows[value_col]
            return future_rows[['date', 'prediction']]

        future_df = recursive_forecast(model, df_processed, periods=forecast_months, feature_cols=X_cols)

        st.subheader("🔮 Future Forecast")
        st.dataframe(future_df)
        st.line_chart(future_df.set_index("date")["prediction"])

        # Download Button
        csv = future_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Forecast Results as CSV",
            data=csv,
            file_name="future_forecast_results.csv",
            mime="text/csv"
        )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.markdown("💡 *Built by Shubojit Dutta — Sales Forecasting ML Dashboard (Streamlit)*")
