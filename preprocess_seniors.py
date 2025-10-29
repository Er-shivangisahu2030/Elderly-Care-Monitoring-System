import pandas as pd
import os

print("="*60)
print("CLEANING SENIORS MONITORING DATASET")
print("="*60)

# Paths
raw_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\raw\Seniors_Monitoring_DataSet_Celsius.csv"
cleaned_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned\Seniors_Monitoring_DataSet_Celsius_cleaned.csv"

# Create cleaned directory if it doesn't exist
os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)

# Load CSV data
print("\n1. Loading data...")
try:
    df = pd.read_csv(raw_path)
    print(f"   Original shape: {df.shape}")
except FileNotFoundError:
    print(f"   ❌ File not found: {raw_path}")
    exit()
except Exception as e:
    print(f"   ❌ Error loading CSV: {e}")
    exit()

# Remove duplicates
print("\n2. Removing duplicates...")
df = df.drop_duplicates()
print(f"   After removing duplicates: {df.shape}")

# Remove missing values
print("\n3. Removing missing values...")
df = df.dropna()
print(f"   After removing missing values: {df.shape}")

# Remove outliers (adjust column names as per your dataset)
print("\n4. Removing outliers...")
if 'temperature' in df.columns:
    df = df[(df['temperature'] >= 35) & (df['temperature'] <= 41)]
    print(f"   After temperature filter: {df.shape}")

if 'heart_rate' in df.columns:
    df = df[(df['heart_rate'] > 40) & (df['heart_rate'] < 140)]
    print(f"   After heart rate filter: {df.shape}")

# Save cleaned CSV
print("\n5. Saving cleaned data...")
df.to_csv(cleaned_path, index=False)
print(f"   ✅ Cleaned data saved to: {cleaned_path}")

print("\n" + "="*60)
print("SENIORS MONITORING CLEANING COMPLETE!")
print("="*60)
