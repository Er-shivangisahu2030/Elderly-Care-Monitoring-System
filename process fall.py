import os
import pandas as pd

# --- Configuration ---
image_base_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\images"
label_base_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\labels"
processed_labels_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed"

os.makedirs(processed_labels_path, exist_ok=True)

for split in ["train", "test"]:
    image_dir = os.path.join(image_base_path, split)
    label_dir = os.path.join(label_base_path, split)

    data_entries = []
    for file_name in os.listdir(image_dir):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_dir, file_name)
        label_file = os.path.splitext(file_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                content = f.read().strip()
                label = int(content) if content.isdigit() else 0
        else:
            label = 0

        data_entries.append({"image_path": img_path, "label": label})

    df = pd.DataFrame(data_entries)
    out_csv = os.path.join(processed_labels_path, f"fall_{split}_labels_clean.csv")
    df.to_csv(out_csv, index=False)
    print(f" Processed {split} dataset → {out_csv}")