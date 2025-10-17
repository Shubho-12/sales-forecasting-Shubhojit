import os
import pandas as pd
from data_prep import load_and_process

# Load and process data
X, y, df_processed = load_and_process('../data/raw/sales.csv')

# Make processed folder if it doesn't exist
os.makedirs('../data/processed', exist_ok=True)

# Save processed data
df_processed.to_csv('../data/processed/sales_processed.csv', index=False)
print("✅ Processed data saved successfully!")

# Optional: verify
df_test = pd.read_csv('../data/processed/sales_processed.csv')
print(df_test.head())
