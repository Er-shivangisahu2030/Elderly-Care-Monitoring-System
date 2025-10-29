import pandas as pd

raw_path = "E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\datasets\\raw\\cardio_train.csv"
cleaned_path = "E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\datasets\\cleaned\\cardio_train_cleaned.csv"

df = pd.read_csv(raw_path)

# Placeholder for now; you will update with thresholds from EDA
df = df.drop_duplicates()
df = df.dropna()
# Example: Adjust these numbers after you analyze outliers from EDA
# df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 250)]
# df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 200)]
# df = df[(df['height'] >= 140) & (df['height'] <= 210)]
# df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]

df.to_csv(cleaned_path, index=False)
print(f"Cleaned data saved to {cleaned_path}")
