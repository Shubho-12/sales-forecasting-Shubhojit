📊 Sales Forecasting ML Dashboard (Streamlit + ML)

Author: Shubojit Dutta
Tech Stack: Python, Pandas, Scikit-learn, Streamlit, Joblib, Matplotlib

🚀 Project Overview

This project is a Machine Learning-powered dashboard for sales forecasting.
It predicts future sales based on historical data, leveraging feature engineering and regression models.
Users can upload their sales data, visualize actual vs predicted performance, and generate future forecasts dynamically.

🎯 Key Features

✅ Data Upload & Processing

Upload your sales CSV file (with date, store, product, units_sold columns).

Automated preprocessing using src/data_prep.py (trend, lags, rolling means).

✅ Model Prediction & Visualization

Uses a trained RandomForestRegressor model (sales_forecast_model.pkl).

Displays actual vs predicted sales with dynamic line charts.

Supports filtering by store or product if available.

✅ Forecasting Future Sales

Recursive multi-step prediction for the next N months (1–36 months).

Visualizes forecast results interactively and provides CSV download.

✅ Performance Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

✅ Model Insights

Displays top feature importances from the trained model for interpretability.

✅ Interactive Dashboard (Streamlit)

Real-time data preview with row slider.

Clean UI optimized for presentation and recruiters.

🧠 Tech Stack
Component	Technology Used
Programming Language	Python 3.x
Machine Learning	scikit-learn
Data Processing	pandas, numpy
Dashboard	Streamlit
Visualization	Matplotlib
Model Storage	Joblib
⚙️ Project Structure
sales-forecasting/
│
├── src/
│   └── data_prep.py              # Data preprocessing & feature engineering
│
├── models/
│   ├── sales_forecast_model.pkl  # Trained ML model
│   └── X_train_columns.pkl       # Model input columns
│
├── app.py                        # Streamlit dashboard app
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── sample_data.csv               # Example input data

🧩 How to Run Locally

Clone the repository:

git clone https://github.com/<your-username>/sales-forecasting-dashboard.git
cd sales-forecasting-dashboard


Create & activate a virtual environment:

python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate  # For Mac/Linux


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


Open your browser:

http://localhost:8501

📈 Example Workflow

Upload your company’s historical sales CSV.

View preprocessed data and model predictions.

Evaluate metrics and visualize actual vs predicted sales.

Generate a 6–12 month sales forecast.

Download forecast results for business planning.

💼 Resume Highlight

Sales Forecasting ML Dashboard (Python, Streamlit, scikit-learn)
Built an interactive ML-powered web dashboard to forecast future sales trends using time-series regression. Implemented automated feature engineering (lags, rolling windows), model evaluation (RMSE, R²), and visualization of results through Streamlit. Deployed trained RandomForestRegressor model for real-time forecasting with downloadable outputs.

🧾 Requirements

Example requirements.txt:

pandas
numpy
scikit-learn
joblib
matplotlib
streamlit

🌟 Future Enhancements

Add model training retrigger via UI

Integrate with a database (PostgreSQL or MongoDB)

Enable deployment on AWS / Render / Hugging Face Spaces

Add authentication and team dashboards

🙌 Acknowledgment

Developed by Shubojit Dutta as a Machine Learning project for sales forecasting and business analytics use cases.