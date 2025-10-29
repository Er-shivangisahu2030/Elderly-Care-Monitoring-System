import os
import pandas as pd

# ===============================
# CONFIGURATION
# ===============================
ORIGINAL_CSV = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\fall_train_labels_clean.csv"   # Update with your original csv
CLEAN_CSV = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\fall_train_labels_c.csv"

def is_valid_row(row):
    img_path = str(row[0]).strip()
    label = row[1]
    if not os.path.exists(img_path):
        print(f"⚠️ Skipping missing image: {img_path}")
        return False
    if pd.isna(label) or label == '':
        print(f"⚠️ Skipping missing or empty label for: {img_path}")
        return False
    return True

# ===============================
# CLEANING
# ===============================
df = pd.read_csv(ORIGINAL_CSV)
clean_df = df[df.apply(is_valid_row, axis=1)].reset_index(drop=True)
print(f"✅ Cleaned samples: {len(clean_df)} / {len(df)}")

# ===============================
# SAVE CLEAN FILE
# ===============================
os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
clean_df.to_csv(CLEAN_CSV, index=False)
print(f"💾 Cleaned CSV saved at: {CLEAN_CSV}")
