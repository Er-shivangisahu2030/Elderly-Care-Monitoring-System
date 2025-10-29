import pandas as pd
import os

# ===============================
# PATHS
# ===============================
cleaned_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\datasets\\cleaned\\Seniors_Monitoring_DataSet_Celsius_cleaned.csv"
processed_path = r"E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\datasets\\processed\\Seniors_Monitoring_DataSet_Celsius_processed.csv"

# Ensure the processed folder exists
os.makedirs(os.path.dirname(processed_path), exist_ok=True)

# ===============================
# LOAD CLEANED CSV
# ===============================
if not os.path.exists(cleaned_path):
    raise FileNotFoundError(f"❌ File not found at: {cleaned_path}")

df = pd.read_csv(cleaned_path)

print("✅ Columns in CSV:", df.columns.tolist())

# ===============================
# PROCESSING
# ===============================
# Convert timestamp to datetime and sort
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
else:
    print("⚠️ Column 'timestamp' not found. Skipping datetime conversion.")

# Compute rolling average of temperature if column exists
if 'temperature' in df.columns:
    df['temp_rollmean'] = df['temperature'].rolling(window=5, min_periods=1).mean()
else:
    print("⚠️ Column 'temperature' not found. Skipping rolling mean computation.")

# ===============================
# SAVE PROCESSED CSV
# ===============================
df.to_csv(processed_path, index=False)
print(f"✅ Processed seniors monitoring dataset saved at: {processed_path}")
