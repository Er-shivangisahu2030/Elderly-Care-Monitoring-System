# ============================================================================
# STEP 1: DATA CLEANING SCRIPT
# ============================================================================
# This script cleans the Seniors Monitoring Dataset
# Author: Elder Care Monitoring System
# Date: October 2025
# ============================================================================

import pandas as pd
import numpy as np
import os

print("=" * 80)
print("SENIOR HEALTH MONITORING SYSTEM - DATA CLEANING")
print("=" * 80)

# Define paths
RAW_DATA_PATH = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\raw\Seniors_Monitoring_DataSet_Celsius.csv"
CLEANED_DATA_PATH = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned\Seniors_Monitoring_Cleaned.csv"

# Create cleaned directory if it doesn't exist
os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)

print("\n[1/6] Loading raw data...")
print(f"Source: {RAW_DATA_PATH}")

# Load the dataset
df = pd.read_csv(RAW_DATA_PATH)
print(f"✓ Data loaded successfully!")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n[2/6] Inspecting data structure...")
print("\nColumn Names and Types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

print("\n[3/6] Checking for missing values...")
missing_values = df.isnull().sum()
print(missing_values)
total_missing = missing_values.sum()
if total_missing == 0:
    print(f"✓ No missing values found!")
else:
    print(f"⚠ Total missing values: {total_missing}")
    print("Handling missing values...")
    df = df.dropna()
    print(f"✓ Missing values removed. New shape: {df.shape}")

print("\n[4/6] Checking for duplicate rows...")
duplicates = df.duplicated().sum()
if duplicates == 0:
    print(f"✓ No duplicate rows found!")
else:
    print(f"⚠ Found {duplicates} duplicate rows")
    df = df.drop_duplicates()
    print(f"✓ Duplicates removed. New shape: {df.shape}")

print("\n[5/6] Analyzing data quality...")
print("\nBasic Statistics:")
print(df.describe())

print("\nTarget Variable Distribution:")
print(df['Driver_State'].value_counts())
print("\nPercentage Distribution:")
print(df['Driver_State'].value_counts(normalize=True) * 100)

# Check for outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

print("\nOutlier Detection (IQR Method):")
numerical_cols = ['Body_Temperature', 'Heart_Rate', 'SPO2']
for col in numerical_cols:
    outlier_count, lower, upper = detect_outliers_iqr(df, col)
    print(f"\n{col}:")
    print(f"  Outliers: {outlier_count}")
    print(f"  Valid range: [{lower:.2f}, {upper:.2f}]")
    print(f"  Actual range: [{df[col].min():.2f}, {df[col].max():.2f}]")

print("\nNote: Outliers are RETAINED as they may represent critical health conditions.")

print("\n[6/6] Saving cleaned data...")
df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"✓ Cleaned data saved to: {CLEANED_DATA_PATH}")

print("\n" + "=" * 80)
print("DATA CLEANING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Output file: Seniors_Monitoring_Cleaned.csv")
print("\nNext Step: Run '2_data_preprocessing.py'")
print("=" * 80)