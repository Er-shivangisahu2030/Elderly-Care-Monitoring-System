import os
import ast
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# ==========================
# Dataset Class
# ==========================
class FallDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, img_size=(256, 256)):
        """
        csv_file: Path to CSV file with columns ['image_name', 'label', 'image_path'] (or similar).
        img_dir: Folder containing images.
        transform: Optional transform (e.g., from torchvision).
        img_size: Target size for resizing images (default: 256x256 to match model).
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found at {img_dir}")

        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

        # Parse and clean label column (assuming binary classification: 0 or 1)
        def parse_label(label_str):
            try:
                # If label is a stringified list [class, x, y, w, h], take the first element
                if isinstance(label_str, str) and label_str.startswith('['):
                    label_list = ast.literal_eval(label_str)
                    return int(label_list[0])  # e.g., 0 for non-fall, 1 for fall
                else:
                    # If it's already an int/float, convert directly
                    return int(float(label_str))
            except (ValueError, SyntaxError):
                print(f"Warning: Invalid label '{label_str}', defaulting to 0")
                return 0  # Default fallback

        if "label" in self.data.columns:
            self.data["label"] = self.data["label"].apply(parse_label)
        else:
            raise KeyError("CSV must contain a column named 'label'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Handle different column names for image path
        img_name = str(row.get("image_name", row.get("path", row.get("image_path", ""))))
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        # Load and resize image to match model input (256x256)
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size, Image.Resampling.BILINEAR)  # Resize to 256x256

        # Convert to tensor: (H, W, C) -> (C, H, W), normalize to [0,1]
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()

        label = torch.tensor(row["label"], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================
# CNN Model
# ==========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Assuming 256x256 input: After two MaxPool2d(2), feature map is 64x64
        # Flattened: 32 * 64 * 64 = 131072
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================
# Training Function
# ==========================
def train_model(csv_file, img_dir, batch_size=8, epochs=5, lr=0.001, save_path=None):
    print("============================================================")
    print("TRAINING FALL DETECTION MODEL (PyTorch)")
    print("============================================================")

    dataset = FallDataset(csv_file, img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Determine num_classes from data (in case it's not always 2)
    unique_labels = dataset.data["label"].unique()
    num_classes = len(unique_labels)
    print(f"Detected {num_classes} classes: {sorted(unique_labels)}")

    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    print("Training complete.")
    
    # Optional: Save the model
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return model

# ==========================
# Main Script
# ==========================
if __name__ == "__main__":
    # Update these paths as needed (currently absolute; consider relative for portability)
    csv_file = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\fall_train_labels_clean.csv"
    img_dir = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\cleaned\fall_dataset\images\train"
    save_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\models\fall_model.pth"  # Optional save

    trained_model = train_model(csv_file, img_dir, batch_size=4, epochs=5, lr=0.001, save_path=save_path)
