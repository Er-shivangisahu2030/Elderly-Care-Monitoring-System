import os
import pandas as pd

# --- Configuration ---
raw_dataset_dir = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\raw"
cleaned_dataset_dir = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned"

# Create output directory
os.makedirs(cleaned_dataset_dir, exist_ok=True)

# Example: combine metadata or clean CSVs (adjust if you have custom format)
for split in ["train", "test"]:
    csv_path = os.path.join(raw_dataset_dir, f"fall_{split}_labels.csv")
    if not os.path.exists(csv_path):
        print(f"  {csv_path} not found, skipping.")
        continue

    df = pd.read_csv(csv_path)

    # Drop duplicates or NaN rows
    df = df.dropna().drop_duplicates()

    # Ensure label column exists
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()
    else:
        raise KeyError("Missing 'label' column in raw data")

    out_path = os.path.join(cleaned_dataset_dir, f"fall_{split}_labels_clean.csv")
    df.to_csv(out_path, index=False)
    print(f" Cleaned {split} data saved to {out_path}")
