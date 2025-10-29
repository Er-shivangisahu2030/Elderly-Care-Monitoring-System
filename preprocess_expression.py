import os
from PIL import Image

# Source and target directories
RAW_BASE = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\raw\exp detection"
CLEANED_BASE = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned\exp detection"
TARGET_SIZE = (128, 128)

def clean_expression_recursive(src_dir, tgt_dir, target_size):
    """Recursively find images, resize, and save keeping subfolder structure."""
    if not os.path.exists(src_dir):
        print(f"❌ Source directory does not exist: {src_dir}")
        return

    total_images = 0
    cleaned_count = 0
    skipped_count = 0

    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        tgt_path = os.path.join(tgt_dir, rel_path)
        os.makedirs(tgt_path, exist_ok=True)

        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                total_images += 1
                src_file = os.path.join(root, fname)
                tgt_file = os.path.join(tgt_path, fname)
                try:
                    with Image.open(src_file) as img:
                        img = img.convert('RGB')
                        img = img.resize(target_size)
                        img.save(tgt_file)
                        cleaned_count += 1
                        if cleaned_count % 50 == 0:
                            print(f"   ✅ {cleaned_count} images cleaned so far...")
                except Exception as e:
                    skipped_count += 1
                    print(f"⚠️ Skipping {src_file}: {e}")

    print("\n================ SUMMARY ================")
    print(f"Total images found: {total_images}")
    print(f"✅ Successfully cleaned: {cleaned_count}")
    print(f"⚠️ Skipped: {skipped_count}")
    print("=========================================")

print("="*60)
print("CLEANING EXPRESSION DETECTION DATASET (recursive, keeps folder structure)")
print("="*60)

clean_expression_recursive(RAW_BASE, CLEANED_BASE, TARGET_SIZE)

print("\n✅ All images processed and saved to:")
print(CLEANED_BASE)
print("="*60)
