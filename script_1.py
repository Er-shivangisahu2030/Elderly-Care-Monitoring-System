
# Create complete training script for fall detection model using PyTorch
training_script = '''
"""
Fall Detection Model Training Script
Elder Care Assistance and Monitoring System
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import ast
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define paths
TRAIN_CSV = r'fall_train_labels_clean.csv'
TEST_CSV = r'fall_test_labels_clean.csv'
MODEL_SAVE_PATH = r'E:\\PROJECT FILE\\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\\models\\fall_detection_model.pth'

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 224
NUM_CLASSES = 3  # Fall, Standing, Sitting

# Class names
CLASS_NAMES = {0: 'Fall', 1: 'Standing', 2: 'Sitting'}

# Custom Dataset Class
class FallDetectionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to csv file with annotations
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
        # Extract class labels
        self.data_frame['class'] = self.data_frame['label'].apply(
            lambda x: int(ast.literal_eval(x)[0])
        )
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.data_frame.iloc[idx]['image_path']
        label = self.data_frame.iloc[idx]['class']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
print("Loading datasets...")
train_dataset = FallDetectionDataset(TRAIN_CSV, transform=train_transform)
test_dataset = FallDetectionDataset(TEST_CSV, transform=test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Define Model using ResNet50
class FallDetectionModel(nn.Module):
    def __init__(self, num_classes=3):
        super(FallDetectionModel, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Initialize model
print("Initializing model...")
model = FallDetectionModel(num_classes=NUM_CLASSES).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Training loop
print("\\nStarting training...")
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f'\\nEpoch {epoch+1}/{NUM_EPOCHS}')
    print('-' * 60)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validate
    val_loss, val_acc, val_preds, val_labels = validate_epoch(model, test_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'class_names': CLASS_NAMES
        }, MODEL_SAVE_PATH)
        print(f'Model saved with validation accuracy: {val_acc:.2f}%')

print("\\n" + "="*60)
print("Training completed!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

# Final evaluation
print("\\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

# Load best model
checkpoint = torch.load(MODEL_SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])

# Get predictions on test set
_, _, test_preds, test_labels = validate_epoch(model, test_loader, criterion, device)

# Classification report
print("\\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)]))

# Confusion matrix
print("\\nConfusion Matrix:")
cm = confusion_matrix(test_labels, test_preds)
print(cm)

print("\\nModel saved at:", MODEL_SAVE_PATH)
print("Training completed successfully!")
'''

# Save the training script
with open('train_fall_detection_model.py', 'w') as f:
    f.write(training_script)

print("Training script created: train_fall_detection_model.py")
print("\nTo train the model, run:")
print("python train_fall_detection_model.py")
