import os
from PIL import Image
import pandas as pd

print("="*60)
print("CLEANING FALL DETECTION DATASET")
print("="*60)

# Base paths
raw_base = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\raw\fall_dataset"
cleaned_base = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned\fall_dataset"

def clean_images(input_dir, output_dir, target_size=(128, 128)):
    """Clean and resize images by attempting to open any file as an image"""
    if not os.path.exists(input_dir):
        print(f"   ⚠️ Input directory does not exist: {input_dir}")
        return 0, 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # List all files in the directory
    all_files = os.listdir(input_dir)
    print(f"   📁 Files in {input_dir}: {all_files}")
    
    cleaned_count = 0
    skipped_count = 0
    
    for fname in all_files:
        img_path = os.path.join(input_dir, fname)
        # Skip if it's a directory
        if os.path.isdir(img_path):
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            # Save with a standard extension if needed, but keep original name
            output_path = os.path.join(output_dir, fname)
            img.save(output_path)
            cleaned_count += 1
            print(f"   ✅ Cleaned: {fname}")
        except Exception as e:
            print(f"   ⚠️ Skipped {fname}: {e}")
            skipped_count += 1
    
    return cleaned_count, skipped_count

def clean_text_labels(input_labels_dir, output_labels_dir, cleaned_images):
    """Clean text label files by filtering lines based on cleaned images"""
    if not os.path.exists(input_labels_dir):
        print(f"   ⚠️ Input directory does not exist: {input_labels_dir}")
        return 0, 0
    
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # List all files in the directory
    all_files = os.listdir(input_labels_dir)
    print(f"   📁 Files in {input_labels_dir}: {all_files}")
    
    # Assume text files end with .txt (adjust if different)
    text_files = [f for f in all_files if f.lower().endswith('.txt')]
    print(f"   🔍 Detected text label files: {text_files}")
    
    cleaned_count = 0
    skipped_count = 0
    
    for fname in text_files:
        try:
            input_path = os.path.join(input_labels_dir, fname)
            output_path = os.path.join(output_labels_dir, f'cleaned_{fname}')
            
            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    # Assume each line starts with image_name followed by space or tab
                    parts = line.strip().split()
                    if parts and parts[0] in cleaned_images:
                        outfile.write(line)
                        cleaned_count += 1  # Count kept lines
            
            print(f"   ✅ Cleaned text file: {fname} -> {output_path}")
        except Exception as e:
            print(f"   ⚠️ Error processing {fname}: {e}")
            skipped_count += 1
    
    return cleaned_count, skipped_count

# Process train and test splits
for split in ['train', 'test']:
    print(f"\n{'='*60}")
    print(f"Processing {split.upper()} split")
    print('='*60)
    
    # Prepare image directories (from images folder)
    input_img_dir = os.path.join(raw_base, 'images', split)
    output_img_dir = os.path.join(cleaned_base, 'images', split)
    
    if not os.path.exists(input_img_dir):
        print(f"   ⚠️ Skipping {split} images: folder does not exist -> {input_img_dir}")
    else:
        # Clean images from images folder
        print(f"\n1. Cleaning {split} images from images folder...")
        cleaned, skipped = clean_images(input_img_dir, output_img_dir)
        print(f"   ✅ Total Cleaned: {cleaned} images")
        print(f"   ⚠️ Total Skipped: {skipped} images")
    
    # Prepare label directories
    input_labels_dir = os.path.join(raw_base, 'labels', split)
    output_labels_dir = os.path.join(cleaned_base, 'labels', split)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    if not os.path.exists(input_labels_dir):
        print(f"   ⚠️ Skipping {split} labels: folder does not exist -> {input_labels_dir}")
        continue
    
    # Clean images from labels folder (attempt to clean any file that can be opened as an image)
    print(f"\n2. Cleaning {split} images from labels folder...")
    cleaned_labels_img, skipped_labels_img = clean_images(input_labels_dir, output_labels_dir)
    print(f"   ✅ Total Cleaned: {cleaned_labels_img} images")
    print(f"   ⚠️ Total Skipped: {skipped_labels_img} images")
    
    # Collect all cleaned images for filtering labels
    cleaned_images = set()
    if os.path.exists(output_img_dir):
        cleaned_images.update(os.listdir(output_img_dir))
    if os.path.exists(output_labels_dir):
        # Only include non-text files as images (since .txt are labels)
        cleaned_images.update([f for f in os.listdir(output_labels_dir) if not f.lower().endswith('.txt')])
    
    # Clean text labels
    print(f"\n3. Cleaning {split} text labels...")
    cleaned_labels, skipped_labels = clean_text_labels(input_labels_dir, output_labels_dir, cleaned_images)
    print(f"   ✅ Total Kept Lines: {cleaned_labels}")
    print(f"   ⚠️ Total Skipped Files: {skipped_labels}")

print("\n" + "="*60)
print("FALL DETECTION CLEANING COMPLETE!")
print("="*60)
