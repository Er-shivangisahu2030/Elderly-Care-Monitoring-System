import pandas as pd
import os

# ===============================
# PATHS
# ===============================
cleaned_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned\cardio_train_cleaned.csv"
processed_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\cardio_train_processed.csv"

# Ensure the processed folder exists
os.makedirs(os.path.dirname(processed_path), exist_ok=True)

# ===============================
# LOAD CLEANED DATA (semicolon-separated)
# ===============================
if not os.path.exists(cleaned_path):
    raise FileNotFoundError(f"❌ Cleaned CSV not found at: {cleaned_path}")

# Explicitly specify separator
df = pd.read_csv(cleaned_path, sep=';')

print("✅ Columns in CSV:", df.columns.tolist())

# ===============================
# FEATURE ENGINEERING
# ===============================
# BMI calculation
if 'weight' in df.columns and 'height' in df.columns:
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
else:
    raise KeyError("❌ Columns 'weight' and/or 'height' not found in CSV!")

# Age in years
if 'age' in df.columns:
    df['age_years'] = df['age'] // 365
else:
    raise KeyError("❌ Column 'age' not found in CSV!")

# One-hot encoding for categorical columns
for col in ['cholesterol', 'gluc']:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# ===============================
# SAVE PROCESSED DATA
# ===============================
df.to_csv(processed_path, index=False)
print(f"✅ Processed cardio train dataset saved at: {processed_path}")
