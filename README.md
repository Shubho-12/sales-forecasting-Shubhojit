# 🧠 Sales Forecasting System (Machine Learning + AWS)

An end-to-end project to predict future sales using Machine Learning.

## 🚀 Project Overview
This project builds a full ML pipeline — from raw data to deployment:
1. Data collection and cleaning
2. Feature engineering
3. Model training (Random Forest / XGBoost)
4. Flask API for prediction
5. AWS EC2 + S3 deployment
6. Optional dashboard (Power BI / Plotly)

## 📁 Folder Structure

## ⚙️ Tech Stack
- Python, Pandas, Scikit-Learn, XGBoost
- Flask (for REST API)
- AWS EC2, S3
- Power BI / Plotly for visualization

## 🧩 How to Run
```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Run Flask API
python src/app.py

Then **save** the file.

---

## 🚫 STEP 3 — Create `.gitignore`

In PyCharm:
1. Right-click project → **New → File**
2. Name it `.gitignore`
3. Paste this:


Save it.

---

## ✅ STEP 4 — Commit and Push to GitHub

Now go back to your terminal and run:

```bash
git add .
git commit -m "Added requirements.txt, README.md, and .gitignore"
git push
