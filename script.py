
import pandas as pd
import ast

# Load the training and test labels
train_df = pd.read_csv(r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\script\exported-assets\fall_train_labels_clean.csv")
test_df = pd.read_csv(r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\script\exported-assets\fall_test_labels_clean.csv")

# Display basic information about the datasets
print("=== TRAINING DATA ===")
print(f"Number of training samples: {len(train_df)}")
print(f"Columns: {train_df.columns.tolist()}")
print("\nFirst few rows:")
print(train_df.head())

print("\n=== TEST DATA ===")
print(f"Number of test samples: {len(test_df)}")
print(f"Columns: {test_df.columns.tolist()}")
print("\nFirst few rows:")
print(test_df.head())

# Parse the label column to understand the structure
# Labels appear to be in format: [class, x_center, y_center, width, height]
sample_label = ast.literal_eval(train_df['label'].iloc[0])
print(f"\n=== LABEL STRUCTURE ===")
print(f"Sample label: {sample_label}")
print(f"Label length: {len(sample_label)}")
print("Format appears to be: [class, x_center, y_center, width, height]")

# Extract class labels from all samples
def extract_class(label_str):
    label = ast.literal_eval(label_str)
    # Get the first element which is the class
    return int(label[0])

train_df['class'] = train_df['label'].apply(extract_class)
test_df['class'] = test_df['label'].apply(extract_class)

print("\n=== CLASS DISTRIBUTION ===")
print("Training set:")
print(train_df['class'].value_counts().sort_index())
print("\nTest set:")
print(test_df['class'].value_counts().sort_index())

# Map class numbers to names
class_names = {0: 'Fall', 1: 'Standing', 2: 'Sitting'}
print("\n=== CLASS MAPPING ===")
for k, v in class_names.items():
    print(f"Class {k}: {v}")
