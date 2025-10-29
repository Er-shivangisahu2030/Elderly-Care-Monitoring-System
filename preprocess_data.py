import os
from torchvision import datasets, transforms

# Define base path
DATASET_DIR = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\raw\fall_dataset"

# Debug: Print directory contents to verify structure
print("Contents of DATASET_DIR:")
if os.path.exists(DATASET_DIR):
    for item in os.listdir(DATASET_DIR):
        print(f"  {item}")
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path):
            print(f"    Subfolders in {item}:")
            for subitem in os.listdir(item_path):
                print(f"      {subitem}")
else:
    print("  DATASET_DIR does not exist!")
    exit()

# Data transforms (preprocessing)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Paths (adjust based on debug output above)
# Assuming structure: fall_dataset/train/fall/, fall_dataset/train/no_fall/
train_dir = os.path.join(DATASET_DIR, "train")
test_dir = os.path.join(DATASET_DIR, "test")

# Alternative paths if train/test are under images/ (uncomment if needed)
# train_dir = os.path.join(DATASET_DIR, "images", "train")
# test_dir = os.path.join(DATASET_DIR, "images", "test")

# Check if paths exist
if not os.path.exists(train_dir):
    print(f"Error: Train directory not found: {train_dir}")
    print("Check the debug output above and adjust the path (e.g., add 'images/' if needed).")
    exit()
if not os.path.exists(test_dir):
    print(f"Error: Test directory not found: {test_dir}")
    print("Check the debug output above and adjust the path.")
    exit()

# Load datasets using ImageFolder (infers labels from subfolders)
train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)

# Map class names to indices (adjust if subfolders are named differently, e.g., '0' and '1')
class_names = ['no_fall', 'fall']  # Assuming subfolders: no_fall (0), fall (1)
train_dataset.class_to_idx = {class_names[i]: i for i in range(len(class_names))}
test_dataset.class_to_idx = {class_names[i]: i for i in range(len(class_names))}

print(f"Preprocessing complete. Train dataset: {len(train_dataset)} samples, Test dataset: {len(test_dataset)} samples.")
print("Proceed to processing.")
