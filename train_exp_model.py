import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random  # For random augmentations

# Optional: For progress bars (install with pip install tqdm)
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ==========================
# Custom Dataset Class
# ==========================
class ExpressionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, augment=False):
        """
        csv_file: Path to CSV with columns ['image_path', 'label']
        img_dir: Base directory for images (if image_path is relative)
        transform: Custom transform function (replaces torchvision)
        augment: Whether to apply augmentations (for training)
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}. Please check the path.")
        
        try:
            self.data = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {e}")
        
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        
        # Validate columns
        required_cols = ['image_path', 'label']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"CSV must have columns: {required_cols}. Found: {list(self.data.columns)}")
        
        # Create label mapping (e.g., 'happy' -> 0, 'sad' -> 1)
        unique_labels = sorted(self.data['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Dataset loaded: {len(self.data)} samples, {len(unique_labels)} classes.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])
        
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            label = self.label_to_idx[row['label']]
            
            if self.transform:
                image = self.transform(image, augment=self.augment)
            
            return image, label
        except Exception as e:
            print(f"Warning: Skipping sample {idx} due to error: {e}")
            # Return a dummy sample to avoid crashes (adjust as needed)
            dummy_image = torch.zeros(3, 128, 128)  # Match expected shape
            dummy_label = 0
            return dummy_image, dummy_label

# ==========================
# Manual Transforms (Replaces torchvision.transforms)
# ==========================
def manual_transform(image, target_size=(128, 128), augment=False, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
    """
    Manual equivalent of torchvision transforms.
    - Resize to target_size
    - Apply augmentations if augment=True (rotation, flip, shift)
    - Convert to tensor and normalize
    """
    try:
        # Resize
        image = image.resize(target_size, Image.BILINEAR)
        
        if augment:
            # Random rotation (0-20 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-20, 20)
                image = image.rotate(angle)
            
            # Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random shift (width/height shift up to 20%)
            if random.random() > 0.5:
                width, height = image.size
                shift_x = int(random.uniform(-0.2, 0.2) * width)
                shift_y = int(random.uniform(-0.2, 0.2) * height)
                image = image.transform(image.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))
        
        # Convert to NumPy array (H, W, C) -> (C, H, W), normalize to [0,1]
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        # Normalize
        image = (image - np.array(normalize_mean)[:, None, None]) / np.array(normalize_std)[:, None, None]
        
        # Convert to PyTorch tensor
        image = torch.from_numpy(image).float()
        
        return image
    except Exception as e:
        print(f"Transform error: {e}. Returning dummy tensor.")
        return torch.zeros(3, 128, 128)  # Dummy fallback

# ==========================
# CNN Model (Equivalent to Keras Sequential)
# ==========================
class ExpressionCNN(nn.Module):
    def __init__(self, num_classes):
        super(ExpressionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),  # 128x128 / 2^3 = 16x16
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================
# Training Function
# ==========================
def train_model(train_csv, test_csv, img_dir, model_path, batch_size=16, epochs=20, patience=5, debug=True):
    print("="*60)
    print("TRAINING EXPRESSION DETECTION MODEL (PYTORCH - NO TORCHVISION)")
    print("="*60)
    
    try:
        # Load datasets with manual transforms
        print("Loading training dataset...")
        train_dataset = ExpressionDataset(train_csv, img_dir, transform=manual_transform, augment=True)
        print("Loading validation dataset...")
        val_dataset = ExpressionDataset(test_csv, img_dir, transform=manual_transform, augment=False)  # No augment for val
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        num_classes = len(train_dataset.label_to_idx)
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {list(train_dataset.label_to_idx.keys())}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Model, loss, optimizer
        device = torch.device('cpu')  # Force CPU to avoid any GPU issues
        print(f"Using device: {device}")
        
        model = ExpressionCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}...")
            # Training
            model.train()
            train_loss = 0.0
            train_batches = len(train_loader)
            for batch_idx, (images, labels) in enumerate(train_loader if not USE_TQDM else tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
                try:
                    images, labels = images.to(device), labels.to(device)
                    
                    # Shape check
                    if images.shape[0] == 0 or images.shape[1:] != (3, 128, 128):
                        print(f"Skipping invalid train batch {batch_idx}: shape {images.shape}")
                        continue
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    if debug and batch_idx % 10 == 0:
                        print(f"  Processed train batch {batch_idx}/{train_batches} - Loss: {loss.item():.4f}")
                
                except Exception as e:
                    print(f"Error in train batch {batch_idx}: {e}")
                    continue
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            val_batches = len(val_loader)
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader if not USE_TQDM else tqdm(val_loader, desc=f"Val Epoch {epoch+1}")):
                    try:
                        images, labels = images.to(device), labels.to(device)
                        
                        if images.shape[0] == 0 or images.shape[1:] != (3, 128, 128):
                            print(f"Skipping invalid val batch {batch_idx}: shape {images.shape}")
                            continue
                        
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, preds = torch.max(outputs, 1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                        
                        if debug and batch_idx % 10 == 0:
                            print(f"  Processed val batch {batch_idx}/{val_batches} - Loss: {loss.item():.4f}")
                    
                    except Exception as e:
                        print(f"Error in val batch {batch_idx}: {e}")
                        continue
            
            train_loss /= train_batches if train_batches > 0 else 1
            val_loss /= val_batches if val_batches > 0 else 1
            val_acc = correct / total if total > 0 else 0
            
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
                print("  Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training complete. Model saved at {model_path}")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Check paths, data, and dependencies.")

# ==========================
# Main Script
# ==========================
if __name__ == "__main__":
    # Paths (update if needed; these are absolute)
    train_csv = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\expression_train_images.csv"
    test_csv = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\expression_test_images.csv"
    img_dir = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned\exp detection"
    model_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\models\expression_detection_model.pth"
    
    # Check paths before starting
    for path, name in [(train_csv, "Train CSV"), (test_csv, "Test CSV"), (img_dir, "Image Dir")]:
        if not (os.path.exists(path) if 'Dir' in name else os.path.isfile(path)):
            print(f"Error: {name} not found at {path}. Exiting.")
            exit(1)
    
    # For testing: Set epochs=1 to run quickly and check for pauses
    train_model(train_csv, test_csv, img_dir, model_path, batch_size=16, epochs=1, patience=5, debug=True)
