import pandas as pd

input_csv = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\fall_train_labels.csv"
output_csv = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\fall_train_labels_clean.csv"

# Read CSV
df = pd.read_csv(input_csv)

# Function to convert label string to list of numbers
def clean_label(label_str):
    # Split by space or comma
    parts = label_str.replace(',', ' ').split()
    # Convert to float or int
    return [float(x) for x in parts]

# Apply to 'label' column
df['label'] = df['label'].apply(clean_label)

# Save cleaned CSV
df.to_csv(output_csv, index=False)
print(f"Cleaned CSV saved at: {output_csv}")
print(df.head())
