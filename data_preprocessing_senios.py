# ============================================================================
# STEP 2: DATA PREPROCESSING SCRIPT
# ============================================================================
# This script preprocesses the cleaned data for model training
# Author: Elder Care Monitoring System
# Date: October 2025
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import os

print("=" * 80)
print("SENIOR HEALTH MONITORING SYSTEM - DATA PREPROCESSING")
print("=" * 80)

# Define paths
CLEANED_DATA_PATH = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned\Seniors_Monitoring_Cleaned.csv"
PROCESSED_DIR = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed"
MODELS_DIR = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\models"

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("\n[1/7] Loading cleaned data...")
print(f"Source: {CLEANED_DATA_PATH}")

# Load cleaned data
df = pd.read_csv(CLEANED_DATA_PATH)
print(f"✓ Data loaded successfully!")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n[2/7] Separating features and target...")
feature_columns = ['Body_Temperature', 'Heart_Rate', 'SPO2']
target_column = 'Driver_State'

X = df[feature_columns]
y = df[target_column]

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

print("\n[3/7] Encoding target variable...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"✓ Target variable encoded!")
print(f"  Classes: {label_encoder.classes_}")
for idx, class_name in enumerate(label_encoder.classes_):
    print(f"    {idx} → {class_name}")

print("\n[4/7] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"✓ Data split completed!")
print(f"  Training set: {X_train.shape[0]} samples ({(X_train.shape[0]/len(X)*100):.1f}%)")
print(f"  Testing set: {X_test.shape[0]} samples ({(X_test.shape[0]/len(X)*100):.1f}%)")

print("\n[5/7] Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled!")
print(f"  Mean: {scaler.mean_}")
print(f"  Std: {scaler.scale_}")

print("\n[6/7] Saving processed datasets...")

# Save training data
train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
train_df['Driver_State_Encoded'] = y_train
train_file = os.path.join(PROCESSED_DIR, 'train_data.csv')
train_df.to_csv(train_file, index=False)
print(f"✓ Training data saved: {train_file}")

# Save testing data
test_df = pd.DataFrame(X_test_scaled, columns=feature_columns)
test_df['Driver_State_Encoded'] = y_test
test_file = os.path.join(PROCESSED_DIR, 'test_data.csv')
test_df.to_csv(test_file, index=False)
print(f"✓ Testing data saved: {test_file}")

# Save original (non-scaled) data for reference
train_original = X_train.copy()
train_original['Driver_State'] = label_encoder.inverse_transform(y_train)
train_original_file = os.path.join(PROCESSED_DIR, 'train_data_original.csv')
train_original.to_csv(train_original_file, index=False)
print(f"✓ Original training data saved: {train_original_file}")

test_original = X_test.copy()
test_original['Driver_State'] = label_encoder.inverse_transform(y_test)
test_original_file = os.path.join(PROCESSED_DIR, 'test_data_original.csv')
test_original.to_csv(test_original_file, index=False)
print(f"✓ Original testing data saved: {test_original_file}")

print("\n[7/7] Saving preprocessing artifacts...")

# Save scaler
scaler_file = os.path.join(MODELS_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_file)
print(f"✓ Scaler saved: {scaler_file}")

# Save label encoder
encoder_file = os.path.join(MODELS_DIR, 'label_encoder.pkl')
joblib.dump(label_encoder, encoder_file)
print(f"✓ Label encoder saved: {encoder_file}")

# Save preprocessing metadata
metadata = {
    'feature_columns': feature_columns,
    'target_column': target_column,
    'classes': list(label_encoder.classes_),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'scaler_mean': list(scaler.mean_) if scaler.mean_ is not None else [],
    'scaler_std': list(scaler.scale_) if scaler.scale_ is not None else []
}

metadata_df = pd.DataFrame([metadata])
metadata_file = os.path.join(PROCESSED_DIR, 'preprocessing_metadata.csv')
metadata_df.to_csv(metadata_file, index=False)
print(f"✓ Metadata saved: {metadata_file}")

print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"Processed files:")
print(f"  - train_data.csv ({len(X_train)} samples)")
print(f"  - test_data.csv ({len(X_test)} samples)")
print(f"  - train_data_original.csv (non-scaled)")
print(f"  - test_data_original.csv (non-scaled)")
print(f"\nPreprocessing artifacts:")
print(f"  - scaler.pkl")
print(f"  - label_encoder.pkl")
print(f"\nNext Step: Run '3_model_training.py'")
print("=" * 80)
#============================================================================
# STEP 3: MODEL TRAINING
# ============================================================================
print("\n" + "=" * 100)
print("STEP 3: MODEL TRAINING")
print("=" * 100)

print("\n[1/7] Initializing Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
print(f"✓ Model configuration:")
print(f"  Algorithm: Random Forest")
print(f"  Trees: 100")
print(f"  Max depth: 10")

print("\n[2/7] Training model on {0} samples...".format(len(X_train_scaled)))
model.fit(X_train_scaled, y_train)
print(f"✓ Model training completed!")

print("\n[3/7] Making predictions on test set...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Model Accuracy: {accuracy * 100:.2f}%")

print("\n[4/7] Calculating feature importance...")
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']*100:.2f}%")

print("\n[5/7] Generating classification report...")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=2)
print("\n" + str(report))

print("\n[6/7] Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\n[7/7] Saving model and metrics...")
# Save model
joblib.dump(model, os.path.join(MODELS_DIR, 'senior_monitoring_model.pkl'))

# Save feature importance
feature_importance.to_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False)

# Save metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
metrics_data = {
    'Metric': ['Accuracy', 'Precision_Avg', 'Recall_Avg', 'F1_Score_Avg', 
            'Training_Samples', 'Testing_Samples', 'Features'],
    'Value': [accuracy, precision, recall, f1,
            len(X_train), len(X_test), len(feature_columns)]
}
pd.DataFrame(metrics_data).to_csv(os.path.join(MODELS_DIR, 'model_metrics.csv'), index=False)

# Save classification report as CSV
report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
pd.DataFrame(report_dict).transpose().to_csv(os.path.join(MODELS_DIR, 'classification_report.csv'))

# Save confusion matrix
pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_).to_csv(
    os.path.join(MODELS_DIR, 'confusion_matrix.csv')
)

print(f"✓ All model files saved successfully")