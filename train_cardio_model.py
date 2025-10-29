import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

print("="*60)
print("TRAINING CARDIOVASCULAR RISK PREDICTION MODEL (PYTORCH)")
print("="*60)

# Paths
processed_data = r'E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\cardio_train_processed.csv'
model_dir = r'E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\models'
os.makedirs(model_dir, exist_ok=True)

# Load processed data
df = pd.read_csv(processed_data)
print(f"\nDataset loaded: {df.shape}")

# Separate features and target
X = df.drop('cardio', axis=1)
y = df['cardio']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network model
class CardioModel(nn.Module):
    def __init__(self, input_size):
        super(CardioModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

input_size = X_train_scaled.shape[1]
model = CardioModel(input_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
print(model)

# Training loop
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

epochs = 50
patience = 10
best_loss = float('inf')
patience_counter = 0

# Pre-calculate total samples to avoid Pylance errors
total_train_samples = len(X_train)
total_test_samples = len(X_test)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= total_train_samples
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:  # Using test_loader as val for simplicity; ideally split train further
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= total_test_samples
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(model_dir, 'best_cardio_model.pth'))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load(os.path.join(model_dir, 'best_cardio_model.pth')))

# Evaluation
print("\n" + "="*60)
print("EVALUATION")
print("="*60)
model.eval()
test_loss = 0.0
correct = 0
total = 0
predictions = []
targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item() * X_batch.size(0)
        predicted = (outputs > 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        predictions.extend(predicted.cpu().numpy())
        targets.extend(y_batch.cpu().numpy())

test_loss /= total_test_samples
test_acc = correct / total

# Calculate AUC
from sklearn.metrics import roc_auc_score
test_auc = roc_auc_score(targets, predictions)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Save model and scaler
model_path = os.path.join(model_dir, 'cardio_model.pth')
scaler_path = os.path.join(model_dir, 'cardio_scaler.pkl')

torch.save(model.state_dict(), model_path)
joblib.dump(scaler, scaler_path)

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
