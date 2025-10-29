import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import numpy as np

# =====================================
# DEVICE SETUP
# =====================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================================
# FILE PATHS (Update with correct ones)
# =====================================
data_path = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\datasets\processed\Seniors_Monitoring_DataSet_Celsius_processed.csv"
model_dir = r"E:\PROJECT FILE\PROJECT 3-ELDER CARE ASSISTENCE AND MONITORING SYSTEM\models"
os.makedirs(model_dir, exist_ok=True)

# =====================================
# LOAD AND PREPROCESS DATA
# =====================================
df = pd.read_csv(data_path)

target = 'Driver_State'
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset!")

# Split features and target
X = df.drop(columns=[target])
y = df[target]

# Encode target labels (e.g., Normal, Fatigue, etc.)
le = LabelEncoder()
y = le.fit_transform(y)

# Check unique values
print(f"Encoded target classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

# =====================================
# CONVERT TO TENSORS
# =====================================
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)  # long for CrossEntropyLoss

X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# =====================================
# DEFINE MODEL (MULTI-CLASS)
# =====================================
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.bn1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No sigmoid — raw logits for CrossEntropyLoss
        return x

num_classes = len(np.unique(y))
model = Net(X_train.shape[1], num_classes).to(device)

# =====================================
# LOSS & OPTIMIZER
# =====================================
criterion = nn.CrossEntropyLoss()  # For multi-class
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================================
# DATALOADER
# =====================================
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =====================================
# TRAINING LOOP
# =====================================
patience = 5
best_loss = float('inf')
epochs_no_improve = 0
best_model_state = None
num_epochs = 40

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss.item():.4f}")

    # Early stopping
    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        best_model_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Restore best model
if best_model_state:
    model.load_state_dict(best_model_state)

# =====================================
# SAVE MODEL AND SCALER
# =====================================
torch.save(model.state_dict(), os.path.join(model_dir, 'seniors_monitoring_model_multiclass.pth'))
joblib.dump(scaler, os.path.join(model_dir, 'seniors_scaler.pkl'))
joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))

print("✅ Model (multi-class), scaler, and label encoder saved successfully!")
