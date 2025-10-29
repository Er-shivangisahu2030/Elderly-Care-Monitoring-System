import os
import pandas as pd

# ============================
# CONFIGURATION
# ============================
base_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned\fall_dataset"
processed_labels_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed"
os.makedirs(processed_labels_path, exist_ok=True)

# ============================
# PROCESS EACH SPLIT
# ============================
for split in ['train', 'test']:
    print(f"\n🔹 Processing split: {split.upper()}")
    labels_folder = os.path.join(base_path, 'labels', split)
    images_dir = os.path.join(base_path, 'images', split)

    if not os.path.exists(labels_folder):
        print(f"⚠️ Labels folder not found: {labels_folder}")
        continue
    if not os.path.exists(images_dir):
        print(f"⚠️ Images folder not found: {images_dir}")
        continue

    # Find all .txt label files (assuming one per image)
    label_files = [f for f in os.listdir(labels_folder) if f.lower().endswith('.txt')]
    if not label_files:
        print(f"❌ No label .txt files found in {labels_folder}")
        continue

    print(f"📄 Found {len(label_files)} label files in {labels_folder}")

    # Get list of valid images (lowercased for matching)
    valid_images = []
    image_to_path = {}
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # Assuming image extensions
                valid_images.append(f.lower())
                image_to_path[f.lower()] = os.path.join(root, f)

    print(f"📁 Total images in folder: {len(valid_images)}")

    # ============================
    # PARSE LABELS FROM INDIVIDUAL .TXT FILES
    # ============================
    image_names = []
    labels = []
    image_paths = []
    skipped_empty = 0
    skipped_no_match = 0

    for label_file in label_files:
        label_file_path = os.path.join(labels_folder, label_file)
        
        # Derive image name from label file (remove .txt and assume same name as image)
        image_name_base = os.path.splitext(label_file)[0]  # e.g., 'image1' from 'image1.txt'
        image_name = image_name_base.lower() + '.jpg'  # Default to .jpg, adjust if needed
        
        # Check if corresponding image exists
        if image_name not in valid_images:
            # Try other extensions if .jpg doesn't match
            for ext in ['.jpeg', '.png', '.bmp', '.tiff']:
                alt_name = image_name_base.lower() + ext
                if alt_name in valid_images:
                    image_name = alt_name
                    break
            else:
                print(f"⚠️ No matching image for label file: {label_file}")
                skipped_no_match += 1
                continue
        
        # Read the label from the .txt file
        try:
            with open(label_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if not content:
                    print(f"⚠️ Empty label file: {label_file}")
                    skipped_empty += 1
                    continue
                # Assuming the content is a single label (e.g., '0' or '1')
                label = content
        except Exception as e:
            print(f"⚠️ Error reading {label_file}: {e}")
            continue
        
        # Append to lists
        image_names.append(image_name)
        labels.append(label)
        image_paths.append(image_to_path[image_name])

    # ============================
    # CREATE DATAFRAME
    # ============================
    df_labels = pd.DataFrame({
        'image_name': image_names,
        'label': labels,
        'image_path': image_paths
    })

    print(f"\n📊 Total labels processed: {len(df_labels)}")
    print(f"⚠️ Skipped empty label files: {skipped_empty}")
    print(f"⚠️ Skipped due to no matching image: {skipped_no_match}")

    if df_labels.empty:
        print(f"❌ No valid labels found for {split}. Skipping.")
        continue

    # ============================
    # SAVE OUTPUT CSV
    # ============================
    out_csv = os.path.join(processed_labels_path, f"fall_{split}_labels.csv")
    df_labels.to_csv(out_csv, index=False)

    print(f"\n✅ Saved processed {split} labels to: {out_csv}")
    print(f"   Records saved: {len(df_labels)}")
    print("-" * 80)
