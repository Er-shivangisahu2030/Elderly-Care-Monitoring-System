import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Define paths (adjusted based on your folder structure: images and labels subfolders, each with train/test)
DATASET_DIR = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\raw\fall_dataset"
MODEL_DIR = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\models"

# Custom Dataset with Cleaning/Validation
class FallDataset(Dataset):
    def __init__(self, img_dir, label_csv, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Load and clean labels CSV
        self.labels = pd.read_csv(label_csv)
        self.labels = self.labels.dropna()  # Drop rows with missing values
        self.labels.iloc[:, 1] = self.labels.iloc[:, 1].astype(int)  # Ensure labels are integers
        
        # Validate and filter valid entries
        valid_entries = []
        for idx in range(len(self.labels)):
            img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
            if os.path.exists(img_name):
                try:
                    # Attempt to open image to check for corruption
                    Image.open(img_name).convert("RGB").close()
                    valid_entries.append(idx)
                except Exception as e:
                    print(f"Skipping corrupted image: {img_name} (Error: {e})")
            else:
                print(f"Image not found: {img_name}")
        self.labels = self.labels.iloc[valid_entries].reset_index(drop=True)
        print(f"Loaded {len(self.labels)} valid samples from {label_csv}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transforms (processing)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Paths for train/test (corrected based on your structure: images/train, labels/train/labels.csv, etc.)
train_img_dir = os.path.join(DATASET_DIR, "images", "train")
train_label_csv = os.path.join(DATASET_DIR, "labels", "train", "labels.csv")
test_img_dir = os.path.join(DATASET_DIR, "images", "test")
test_label_csv = os.path.join(DATASET_DIR, "labels", "test", "labels.csv")

# Check if paths exist
if not os.path.exists(train_img_dir):
    raise FileNotFoundError(f"Train images directory not found: {train_img_dir}")
if not os.path.exists(train_label_csv):
    raise FileNotFoundError(f"Train labels CSV not found: {train_label_csv}")
if not os.path.exists(test_img_dir):
    raise FileNotFoundError(f"Test images directory not found: {test_img_dir}")
if not os.path.exists(test_label_csv):
    raise FileNotFoundError(f"Test labels CSV not found: {test_label_csv}")

# Load datasets
train_dataset = FallDataset(train_img_dir, train_label_csv, transform=data_transforms)
test_dataset = FallDataset(test_img_dir, test_label_csv, transform=data_transforms)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Device and Model (Modeling)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
except AttributeError:
    model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification: 0=no fall, 1=fall
model = model.to(device)

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
epochs = 10
print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluation
print("Evaluating on test set...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Save Model
os.makedirs(MODEL_DIR, exist_ok=True)
output_model_path = os.path.join(MODEL_DIR, "fall_detection_resnet18.pt")
torch.save(model.state_dict(), output_model_path)
print(f"Model saved to {output_model_path}")
